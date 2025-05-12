import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import importlib
import logging
from contextlib import nullcontext

from src.dataset import get_dataloader
from src.model import get_model
from src.utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logger, calculate_metrics
from src.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
from configs.base_config import BaseConfig

logger = None

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, current_epoch_config, current_epoch_num, global_logger):
    """训练一个 epoch"""
    model.train() # 设置模型为训练模式
    loss_meter = AverageMeter()
    optimizer.zero_grad() # 在 epoch 开始时或累积步数开始时清零梯度

    sdp_context = nullcontext() # 默认为空上下文
    if current_epoch_config.ENABLE_SDP_ATTENTION and hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0": # 检查 torch.nn.attention
        try:
            # enable_flash=True: 优先使用 FlashAttention
            # enable_math=True: 使用基于数学库的 scaled dot product attention (通常也很快)
            # enable_mem_efficient=True: 使用内存优化版本 (如果 Flash 和 Math 都不可用)
            # 更新为新的 API
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
            if current_epoch_num == 0: # 仅在第一个epoch打印一次
                 global_logger.info("已尝试启用 PyTorch SDPA (Flash/Math/MemEfficient Attention) via torch.nn.attention.sdpa_kernel")
        except Exception as e:
            if current_epoch_num == 0:
                global_logger.warning(f"尝试启用 SDPA (torch.nn.attention.sdpa_kernel) 时出错: {e}，将使用标准注意力实现")
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num} [训练]", leave=False)
    with sdp_context: # 应用SDPA上下文
        for i, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(current_epoch_config.DEVICE), targets.to(current_epoch_config.DEVICE)

            # 混合精度训练 (autocast)
            with torch.amp.autocast('cuda',enabled=(scaler is not None)):
                outputs = model(images) # 前向传播
                loss = criterion(outputs, targets) # 计算损失
            
            # 梯度累积
            loss = loss / current_epoch_config.ACCUMULATION_STEPS

            if scaler: # 如果使用混合精度
                scaler.scale(loss).backward() # 反向传播 (缩放梯度)
            else:
                loss.backward() # 反向传播

            # 每 ACCUMULATION_STEPS 步更新一次权重
            if (i + 1) % current_epoch_config.ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer) # 执行优化步骤 (自动反缩放梯度)
                    scaler.update() # 更新缩放器状态
                else:
                    optimizer.step() # 执行优化步骤
                optimizer.zero_grad() # 清零梯度，为下一次累积做准备

            loss_meter.update(loss.item() * current_epoch_config.ACCUMULATION_STEPS, images.size(0))
            progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

            # (可选) 定期记录更详细的日志
            # if (i + 1) % (100 * current_epoch_config.ACCUMULATION_STEPS) == 0:
            #     global_logger.info(f"Epoch {current_epoch_num} Batch {(i+1)//current_epoch_config.ACCUMULATION_STEPS}/{len(dataloader)//current_epoch_config.ACCUMULATION_STEPS} Train Loss: {loss_meter.avg:.4f}")
    
    global_logger.info(f"Epoch {current_epoch_num} [训练] 平均损失: {loss_meter.avg:.4f}")
    return loss_meter.avg


def validate_one_epoch(model, dataloader, criterion, current_epoch_config, current_epoch_num, global_logger):
    """验证一个 epoch"""
    model.eval() # 设置模型为评估模式
    loss_meter = AverageMeter()
    all_preds_sigmoid = [] # 存储所有sigmoid后的预测概率
    all_targets = [] # 存储所有真实标签

    sdp_context = nullcontext()
    if current_epoch_config.ENABLE_SDP_ATTENTION and hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0": # 检查 torch.nn.attention
        try:
            # 更新为新的 API
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
        except Exception:
            pass

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num} [验证]", leave=False)
    with sdp_context:
        with torch.no_grad(): # 验证时不需要计算梯度
            for images, targets in progress_bar:
                images, targets = images.to(current_epoch_config.DEVICE), targets.to(current_epoch_config.DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss_meter.update(loss.item(), images.size(0))
                
                preds_sigmoid = torch.sigmoid(outputs) # 获取概率值
                all_preds_sigmoid.append(preds_sigmoid.cpu())
                all_targets.append(targets.cpu())
                progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    # 将所有批次的预测和标签连接起来
    all_preds_sigmoid = torch.cat(all_preds_sigmoid)
    all_targets = torch.cat(all_targets)
    
    # 计算评估指标（使用配置的阈值）
    metrics = calculate_metrics(all_preds_sigmoid, all_targets, threshold=current_epoch_config.INFERENCE_THRESHOLD)
    
    # 如果所有指标为零，尝试不同的阈值
    if metrics['f1_micro'] == 0 and metrics['f1_macro'] == 0 and metrics['exact_match_ratio'] == 0:
        global_logger.info(f"Epoch {current_epoch_num} [验证] 使用默认阈值 {current_epoch_config.INFERENCE_THRESHOLD} 的所有指标为零，尝试其他阈值...")
        best_f1_micro = 0
        best_threshold = current_epoch_config.INFERENCE_THRESHOLD
        
        # 尝试一系列阈值，更细粒度的低阈值范围
        for threshold in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            temp_metrics = calculate_metrics(all_preds_sigmoid, all_targets, threshold=threshold)
            if temp_metrics['f1_micro'] > best_f1_micro:
                best_f1_micro = temp_metrics['f1_micro']
                best_threshold = threshold
                metrics = temp_metrics
        
        if best_f1_micro > 0:
            global_logger.info(f"Epoch {current_epoch_num} [验证] 找到更好的阈值 {best_threshold}，F1-Micro: {best_f1_micro:.4f}")
            # 如果发现了一个明显更好的阈值，考虑更新配置
            if best_threshold != current_epoch_config.INFERENCE_THRESHOLD and best_f1_micro > 0.05:
                global_logger.info(f"Epoch {current_epoch_num} [验证] 将推理阈值从 {current_epoch_config.INFERENCE_THRESHOLD} 更新为 {best_threshold}")
                current_epoch_config.INFERENCE_THRESHOLD = best_threshold
        else:
            global_logger.warning(f"Epoch {current_epoch_num} [验证] 所有尝试的阈值都导致零指标，这可能表明模型预测或数据集存在问题")
    
    global_logger.info(f"Epoch {current_epoch_num} [验证] 平均损失: {loss_meter.avg:.4f}")
    global_logger.info(f"Epoch {current_epoch_num} [验证] F1-Micro: {metrics['f1_micro']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}, mAP-Macro: {metrics['mAP_macro']:.4f}, ExactMatch: {metrics['exact_match_ratio']:.4f}")
    
    return loss_meter.avg, metrics['f1_micro']


def main(args):
    global logger # 声明使用全局 logger
    
    # 1. 加载当前运行的配置
    # config_name 例如 'example_swin_config.py' -> 'example_swin_config'
    config_module_name = args.config_name.replace('.py', '')
    try:
        current_config_module = importlib.import_module(f"configs.{config_module_name}")
        CurrentConfigClass = getattr(current_config_module, args.config_class_name)
        current_run_config = CurrentConfigClass() # 实例化当前运行的配置对象
    except ModuleNotFoundError:
        print(f"错误: 配置文件模块 'configs.{config_module_name}' 未找到")
        return
    except AttributeError:
        print(f"错误: 在配置文件模块 'configs.{config_module_name}' 中未找到配置类 '{args.config_class_name}'")
        return

    # 2. 设置日志记录器
    logger = setup_logger(current_run_config.LOG_DIR, args.run_name)
    logger.info(f"使用配置: {args.config_name} (类: {args.config_class_name})")
    logger.info(f"运行名称: {args.run_name}")
    logger.info(f"运行设备: {current_run_config.DEVICE}")
    if current_run_config.ENABLE_SDP_ATTENTION and torch.__version__ < "2.0.0":
        logger.warning("配置中 ENABLE_SDP_ATTENTION 为 True, 但 PyTorch 版本低于 2.0.0，SDPA 可能不可用")


    # 3. 准备数据加载器和词汇表 (针对当前运行)
    #    对于从头训练或微调一个新数据集，词汇表将基于 current_run_config.SELECTED_TAGS_CSV 构建
    #    对于继续训练，词汇表将从检查点加载，并覆盖这里的初始构建
    logger.info("正在初始化当前运行的数据加载器和词汇表...")
    train_loader, current_tags_list, current_tag_to_idx = get_dataloader(current_run_config, mode="train")
    # current_run_config.NUM_CLASSES 会在 get_dataloader -> DanbooruDataset 中被动态设置或验证
    logger.info(f"当前运行的目标词汇表大小 (NUM_CLASSES): {current_run_config.NUM_CLASSES} 个标签")
    val_loader, _, _ = get_dataloader(current_run_config, mode="val", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx)


    # 4. 模型初始化、检查点加载与适配
    model = None
    optimizer = None
    scheduler = None
    start_epoch = 0
    best_metric_val = 0.0 # 用于跟踪最佳模型的指标值 (越高越好)
    
    # 用于微调时存储从旧检查点加载的信息
    old_ckpt_config_dict = None
    old_ckpt_model_state_dict = None
    old_ckpt_tags_list = None
    old_ckpt_tag_to_idx = None

    # 模式优先级: 继续训练 (resume) > 微调 (finetune) > 从头训练
    if args.resume_checkpoint:
        # 继续训练 (Resume)
        logger.info(f"准备从检查点继续训练: {args.resume_checkpoint}")
        if not os.path.isfile(args.resume_checkpoint):
            logger.error(f"错误: 用于继续训练的检查点文件未找到于 {args.resume_checkpoint}"); return

        # 1. 加载检查点中的词汇表和配置，这些信息将决定模型结构
        #    先加载到CPU，避免GPU显存问题，后续再移到目标设备
        temp_ckpt_data = torch.load(args.resume_checkpoint, map_location='cpu')
        
        resumed_tags_list = temp_ckpt_data.get('tags_list')
        resumed_tag_to_idx = temp_ckpt_data.get('tag_to_idx')
        resumed_config_dict = temp_ckpt_data.get('config_dict')

        if not resumed_tags_list or not resumed_config_dict:
            logger.error("错误: 继续训练的检查点中缺少词汇表或配置信息"); return
        
        # 2. 使用检查点中的词汇表和配置更新当前运行的设置
        current_tags_list = resumed_tags_list
        current_tag_to_idx = resumed_tag_to_idx
        current_run_config = BaseConfig.from_dict(resumed_config_dict) # 用检查点的配置覆盖当前配置
        current_run_config.DEVICE = args.device if hasattr(args, 'device') and args.device else \
                                   ("cuda" if torch.cuda.is_available() else "cpu") # 允许命令行覆盖设备
        logger.info(f"已从检查点加载配置，NUM_CLASSES 将为 {len(current_tags_list)}")
        
        # 3. 重新初始化数据加载器 (使用从检查点加载的词汇表)
        train_loader, _, _ = get_dataloader(current_run_config, mode="train", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx)
        val_loader, _, _ = get_dataloader(current_run_config, mode="val", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx)

        # 4. 初始化模型 (使用检查点中的 NUM_CLASSES)
        #    此时 PRETRAINED 应为 False，因为权重将从检查点加载
        current_run_config.PRETRAINED = False 
        model = get_model(current_run_config) # 不传入旧检查点信息，因为这是标准的权重加载

        # 5. 初始化优化器和调度器 (结构必须在加载状态之前定义)
        #    学习率等参数从 current_run_config (即检查点中的配置) 获取
        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        # ... 其他优化器
        
        if current_run_config.LR_SCHEDULER == "CosineAnnealingLR":
            # 确保 T_max 是基于剩余的 epochs 还是总 epochs (通常是总 epochs)
            # 如果是从检查点恢复，T_max 应该与初次训练时一致
            t_max_for_resume = current_run_config.LR_SCHEDULER_PARAMS.get('T_max', current_run_config.EPOCHS)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_for_resume, 
                                                             eta_min=current_run_config.LR_SCHEDULER_PARAMS.get('eta_min', 1e-6))
        # ... 其他调度器

        # 6. 加载模型、优化器、调度器的状态
        start_epoch, _, _, _ = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, current_run_config.DEVICE,
            require_optimizer_scheduler=True # 继续训练时，通常要求优化器和调度器状态存在
        )
        logger.info(f"继续训练模式：模型、优化器、调度器状态已加载将从 Epoch {start_epoch} 开始")

    elif args.finetune_checkpoint:
        # 微调 (Finetune)
        logger.info(f"准备从检查点进行微调: {args.finetune_checkpoint}")
        if not os.path.isfile(args.finetune_checkpoint):
            logger.error(f"错误: 用于微调的检查点文件未找到于 {args.finetune_checkpoint}"); return

        # 1. 加载旧检查点中的模型权重、配置和词汇表
        ckpt_data = torch.load(args.finetune_checkpoint, map_location='cpu')
        old_ckpt_model_state_dict = ckpt_data.get('model_state_dict')
        old_ckpt_config_dict = ckpt_data.get('config_dict')
        old_ckpt_tags_list = ckpt_data.get('tags_list')
        old_ckpt_tag_to_idx = ckpt_data.get('tag_to_idx')

        if not old_ckpt_model_state_dict or not old_ckpt_config_dict or not old_ckpt_tags_list:
            logger.error("错误: 微调检查点中缺少模型权重、配置或词汇表信息"); return
        
        # 2. 调用 get_model 进行模型创建和适配
        #    get_model 内部会处理 backbone 权重加载和分类器权重迁移
        #    此时 current_run_config 包含的是新运行的配置 (例如新的学习率、epoch数)
        #    current_tags_list 是新数据集的词汇表
        #    old_ckpt_... 是旧模型的词汇表和配置
        current_run_config.PRETRAINED = False # 因为我们从自己的检查点加载 backbone
        model = get_model(
            current_run_config,
            old_checkpoint_config_dict=old_ckpt_config_dict,
            old_checkpoint_state_dict=old_ckpt_model_state_dict,
            current_tags_list=current_tags_list, current_tag_to_idx=current_tag_to_idx,
            old_tags_list_from_ckpt=old_ckpt_tags_list, old_tag_to_idx_from_ckpt=old_ckpt_tag_to_idx
        )
        # 微调时，通常重新初始化优化器和调度器，使用当前运行配置中的学习率等参数
        start_epoch = 0 # 微调从 epoch 0 开始计数 (相对于本次微调任务)
        logger.info(f"微调模式：模型已创建/适配，优化器和调度器将重新初始化")

    else:
        # 从头训练
        logger.info("从头开始训练 (或使用 ImageNet 预训练权重)")
        # current_run_config.PRETRAINED 控制是否加载 ImageNet 权重
        model = get_model(current_run_config)
        start_epoch = 0
    
    # 5. 初始化优化器和调度器 (如果尚未在 resume 分支中初始化)
    if optimizer is None: # 对于从头训练或微调
        # 参数来自 current_run_config
        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        else:
            logger.error(f"不支持的优化器: {current_run_config.OPTIMIZER}"); return
        logger.info(f"优化器 {current_run_config.OPTIMIZER} 已初始化，学习率: {current_run_config.LEARNING_RATE}")

    if scheduler is None: # 对于从头训练或微调
        if current_run_config.LR_SCHEDULER == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=current_run_config.LR_SCHEDULER_PARAMS.get('T_max', current_run_config.EPOCHS), 
                                                             eta_min=current_run_config.LR_SCHEDULER_PARAMS.get('eta_min', 1e-6))
        elif current_run_config.LR_SCHEDULER == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=current_run_config.LR_SCHEDULER_PARAMS.get('step_size', 10), 
                                                  gamma=current_run_config.LR_SCHEDULER_PARAMS.get('gamma', 0.1))
        elif current_run_config.LR_SCHEDULER == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='max', # 通常基于验证集指标，越高越好
                                                             factor=current_run_config.LR_SCHEDULER_PARAMS.get('factor', 0.1),
                                                             patience=current_run_config.LR_SCHEDULER_PARAMS.get('patience', 5),
                                                             verbose=True)
        # 如果 LR_SCHEDULER 为 None 或其他不支持的值，则 scheduler 保持为 None
        if scheduler:
            logger.info(f"学习率调度器 {current_run_config.LR_SCHEDULER} 已初始化")
        else:
            logger.info("未使用学习率调度器")


    # 6. 定义损失函数和混合精度缩放器
    if current_run_config.LOSS_FN == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "AsymmetricLoss":
        criterion = AsymmetricLoss(**current_run_config.LOSS_FN_PARAMS).to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "WeightedBCEWithLogitsLoss":
        # 此处需要计算 pos_weight，通常基于训练数据中各类别的比例
        # pos_weight = (num_negative_samples / num_positive_samples) for each class
        pos_weight_tensor = None 
        if "pos_weight" in current_run_config.LOSS_FN_PARAMS:
            pos_weight_tensor = torch.tensor(current_run_config.LOSS_FN_PARAMS["pos_weight"], device=current_run_config.DEVICE)
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(current_run_config.DEVICE)
    else:
        logger.error(f"不支持的损失函数: {current_run_config.LOSS_FN}"); return
    logger.info(f"损失函数: {current_run_config.LOSS_FN}")

    # 混合精度训练的梯度缩放器
    scaler = None
    if current_run_config.DEVICE == "cuda" and torch.cuda.is_available():
        try:
            scaler = torch.amp.GradScaler('cuda')
            logger.info("已启用混合精度训练 (GradScaler)")
        except AttributeError:
            logger.warning("torch.amp.GradScaler('cuda') 不可用，尝试使用旧版 torch.cuda.amp.GradScaler()")
            try:
                scaler = torch.amp.GradScaler('cuda')
                logger.info("已启用混合精度训练 (旧版 GradScaler)")
            except AttributeError:
                logger.error("无法初始化 GradScaler，混合精度训练将不可用")
                scaler = None

    elif current_run_config.DEVICE == "cuda" and not torch.cuda.is_available():
        logger.warning("配置设备为 CUDA，但 CUDA 不可用，将使用 CPU，混合精度已禁用")
        current_run_config.DEVICE = "cpu" # 强制回退到 CPU
        model.to("cpu") # 确保模型在CPU上
    else: # CPU 模式
        logger.info("使用 CPU 进行训练，混合精度已禁用")


    # 7. 训练循环
    logger.info(f"开始训练，从 Epoch {start_epoch} 到 {current_run_config.EPOCHS -1}")
    logger.info(f"模型: {current_run_config.MODEL_NAME}, 目标类别数: {current_run_config.NUM_CLASSES}")
    logger.info(f"批次大小: {current_run_config.BATCH_SIZE}, 梯度累积步数: {current_run_config.ACCUMULATION_STEPS}")

    for epoch in range(start_epoch, current_run_config.EPOCHS):
        logger.info(f"Epoch {epoch}/{current_run_config.EPOCHS - 1}")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, current_run_config, epoch, logger)
        
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, current_run_config, epoch, logger)

        # 更新学习率调度器
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metric) # ReduceLROnPlateau 需要一个指标来判断
            else:
                scheduler.step() # 其他调度器在每个 epoch 后更新

        current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
        logger.info(f"Epoch {epoch} 完成，当前学习率: {current_lr:.7f}")

        # 保存检查点
        is_best = val_metric > best_metric_val
        if is_best:
            best_metric_val = val_metric
            logger.info(f"在 Epoch {epoch} 发现新的最佳验证指标: {best_metric_val:.4f}")
        
        checkpoint_name_prefix = args.run_name # 使用运行名称作为检查点文件名的前缀
        
        if current_run_config.SAVE_BEST_ONLY:
            if is_best:
                save_checkpoint(epoch, model, optimizer, scheduler, current_run_config, 
                                current_tags_list, current_tag_to_idx,
                                filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True, 
                                output_dir=current_run_config.OUTPUT_DIR)
        else: # 如果不只保存最佳，则按间隔保存，并且如果当前是最佳也保存
            save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                            current_tags_list, current_tag_to_idx,
                            filename_prefix=f"{checkpoint_name_prefix}_epoch_{epoch}", is_best_marker_for_log=False, # 为每个epoch保存一个文件
                            output_dir=current_run_config.OUTPUT_DIR)
            if is_best: # 如果不是best_only模式，但当前是最佳，也保存一个best标记的文件
                 save_checkpoint(epoch, model, optimizer, scheduler, current_run_config, 
                                current_tags_list, current_tag_to_idx,
                                filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True, 
                                output_dir=current_run_config.OUTPUT_DIR)

        # 始终保存最新的检查点，方便中断后快速恢复 (覆盖式保存)
        save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                        current_tags_list, current_tag_to_idx,
                        filename_prefix=f"{checkpoint_name_prefix}_last", is_best_marker_for_log=False,
                        output_dir=current_run_config.OUTPUT_DIR)

    logger.info("训练完成")
    logger.info(f"最佳验证指标 (F1-Micro): {best_metric_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 图像多标签分类训练脚本")
    parser.add_argument("--config_name", type=str, default="eva02_L_clip336_merged2b_config.py", 
                        help="配置文件名 (例如: example_swin_config.py)，位于 configs/ 目录下")
    parser.add_argument("--config_class_name", type=str, default="Eva02LargeClip336Merged2BConfig", 
                        help="配置文件中配置类的名称 (例如: ExampleSwinConfig)")
    parser.add_argument("--run_name", type=str, default="danbooru_experiment", 
                        help="本次运行的名称，用于日志和模型保存")
    
    # 模式选择参数
    parser.add_argument("--resume_checkpoint", type=str, default=None, 
                        help="从指定的检查点路径继续训练 (加载模型、优化器、调度器状态)")
    parser.add_argument("--finetune_checkpoint", type=str, default=None, 
                        help="从指定的检查点路径进行微调 (加载模型 backbone 权重，可适配新标签集，重新初始化优化器/调度器)")
    
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'],
                        help="覆盖配置文件中的 DEVICE 设置 (例如，在无GPU环境强制使用cpu)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if args.resume_checkpoint and args.finetune_checkpoint:
        logging.error("错误: --resume_checkpoint 和 --finetune_checkpoint 参数不能同时指定")
    else:
        main(args)
