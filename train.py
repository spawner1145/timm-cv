import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import importlib
import logging
from contextlib import nullcontext # 保持 nullcontext

from src.dataset import get_dataloader # 假设 get_dataloader 内部处理单卡/多卡情况或可以适配
from src.model import get_model
from src.utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logger, calculate_metrics
from src.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
from configs.base_config import BaseConfig

logger = None # 全局 logger

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, current_epoch_config, current_epoch_num, global_logger):
    """训练一个 epoch"""
    model.train() # 设置模型为训练模式
    loss_meter = AverageMeter()
    optimizer.zero_grad() # 在 epoch 开始时或累积步数开始时清零梯度

    sdp_context = nullcontext() # 默认为空上下文
    if current_epoch_config.ENABLE_SDP_ATTENTION and hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0": # 检查 torch.nn.attention
        try:
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
            # MODIFICATION: Changed 'cuda' to current_epoch_config.DEVICE for autocast device_type
            # and check if device is 'cuda' for enabling.
            autocast_device_type = current_epoch_config.DEVICE if 'cuda' in str(current_epoch_config.DEVICE) else 'cpu'
            is_cuda_enabled_for_amp = (scaler is not None and 'cuda' in str(current_epoch_config.DEVICE))

            with torch.amp.autocast(device_type=autocast_device_type, enabled=is_cuda_enabled_for_amp):
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
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
        except Exception:
            pass

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num} [验证]", leave=False)

    # --- MODIFICATION START: Determine device type for autocast and enable if cuda ---
    autocast_device_type = current_epoch_config.DEVICE if 'cuda' in str(current_epoch_config.DEVICE) else 'cpu'
    is_cuda_for_val_amp = ('cuda' in str(current_epoch_config.DEVICE))
    # --- MODIFICATION END ---

    with sdp_context:
        with torch.no_grad(): # 验证时不需要计算梯度
            # --- MODIFICATION START: Add autocast context for validation ---
            with torch.amp.autocast(device_type=autocast_device_type, enabled=is_cuda_for_val_amp):
            # --- MODIFICATION END ---
                for images, targets_batch in progress_bar: # Renamed targets to targets_batch for clarity
                    images, targets_gpu = images.to(current_epoch_config.DEVICE), targets_batch.to(current_epoch_config.DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, targets_gpu)
                    loss_meter.update(loss.item(), images.size(0))

                    preds_sigmoid = torch.sigmoid(outputs) # 获取概率值
                    all_preds_sigmoid.append(preds_sigmoid.cpu())
                    all_targets.append(targets_batch.cpu()) # Use targets_batch here
                    progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    if not all_preds_sigmoid: # 检查列表是否为空
        global_logger.error(f"Epoch {current_epoch_num} [验证] 未收集到任何预测，无法计算指标。")
        return loss_meter.avg, 0.0


    all_preds_sigmoid = torch.cat(all_preds_sigmoid)
    all_targets = torch.cat(all_targets)

    metrics = calculate_metrics(all_preds_sigmoid, all_targets, threshold=current_epoch_config.INFERENCE_THRESHOLD)

    if metrics['f1_micro'] == 0 and metrics['f1_macro'] == 0 and metrics['exact_match_ratio'] == 0:
        global_logger.info(f"Epoch {current_epoch_num} [验证] 使用默认阈值 {current_epoch_config.INFERENCE_THRESHOLD} 的所有指标为零，尝试其他阈值...")
        best_f1_micro = 0
        best_threshold = current_epoch_config.INFERENCE_THRESHOLD

        for threshold_val in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]: # Renamed threshold to threshold_val
            temp_metrics = calculate_metrics(all_preds_sigmoid, all_targets, threshold=threshold_val)
            if temp_metrics['f1_micro'] > best_f1_micro:
                best_f1_micro = temp_metrics['f1_micro']
                best_threshold = threshold_val
                metrics = temp_metrics

        if best_f1_micro > 0:
            global_logger.info(f"Epoch {current_epoch_num} [验证] 找到更好的阈值 {best_threshold}，F1-Micro: {best_f1_micro:.4f}")
            if best_threshold != current_epoch_config.INFERENCE_THRESHOLD and best_f1_micro > 0.05: # 仅在显著改善时更新
                global_logger.info(f"Epoch {current_epoch_num} [验证] 将推理阈值从 {current_epoch_config.INFERENCE_THRESHOLD} 更新为 {best_threshold}")
                current_epoch_config.INFERENCE_THRESHOLD = best_threshold
        else:
            global_logger.warning(f"Epoch {current_epoch_num} [验证] 所有尝试的阈值都导致零指标，这可能表明模型预测或数据集存在问题")

    global_logger.info(f"Epoch {current_epoch_num} [验证] 平均损失: {loss_meter.avg:.4f}")
    global_logger.info(f"Epoch {current_epoch_num} [验证] F1-Micro: {metrics['f1_micro']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}, mAP-Macro: {metrics['mAP_macro']:.4f}, ExactMatch: {metrics['exact_match_ratio']:.4f}")

    return loss_meter.avg, metrics['f1_micro']


def main(args):
    global logger

    config_module_name = args.config_name.replace('.py', '')
    try:
        current_config_module = importlib.import_module(f"configs.{config_module_name}")
        CurrentConfigClass = getattr(current_config_module, args.config_class_name)
        current_run_config = CurrentConfigClass()
    except ModuleNotFoundError:
        print(f"错误: 配置文件模块 'configs.{config_module_name}' 未找到")
        return
    except AttributeError:
        print(f"错误: 在配置文件模块 'configs.{config_module_name}' 中未找到配置类 '{args.config_class_name}'")
        return

    # 命令行参数可以覆盖配置文件中的DEVICE设置
    if args.device:
        current_run_config.DEVICE = args.device
    elif not torch.cuda.is_available() and "cuda" in current_run_config.DEVICE:
        print("警告: 配置文件请求CUDA但CUDA不可用，将使用CPU。")
        current_run_config.DEVICE = "cpu"


    logger = setup_logger(current_run_config.LOG_DIR, args.run_name) # setup_logger 不再需要 rank
    logger.info(f"使用配置: {args.config_name} (类: {args.config_class_name})")
    logger.info(f"运行名称: {args.run_name}")
    logger.info(f"运行设备: {current_run_config.DEVICE}")
    if current_run_config.ENABLE_SDP_ATTENTION and torch.__version__ < "2.0.0":
        logger.warning("配置中 ENABLE_SDP_ATTENTION 为 True, 但 PyTorch 版本低于 2.0.0，SDPA 可能不可用")


    logger.info("正在初始化当前运行的数据加载器和词汇表...")
    # For single card, rank=0, world_size=1
    train_loader, current_tags_list, current_tag_to_idx = get_dataloader(current_run_config, mode="train", rank=0, world_size=1)
    logger.info(f"当前运行的目标词汇表大小 (NUM_CLASSES): {current_run_config.NUM_CLASSES} 个标签")
    val_loader, _, _ = get_dataloader(current_run_config, mode="val", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx, rank=0, world_size=1)


    model = None
    optimizer = None
    scheduler = None
    start_epoch = 0
    best_metric_val = 0.0

    old_ckpt_config_dict = None
    old_ckpt_model_state_dict = None
    old_ckpt_tags_list = None
    old_ckpt_tag_to_idx = None

    if args.resume_checkpoint:
        logger.info(f"准备从检查点继续训练: {args.resume_checkpoint}")
        if not os.path.isfile(args.resume_checkpoint):
            logger.error(f"错误: 用于继续训练的检查点文件未找到于 {args.resume_checkpoint}"); return

        temp_ckpt_data = torch.load(args.resume_checkpoint, map_location='cpu')
        resumed_tags_list = temp_ckpt_data.get('tags_list')
        resumed_tag_to_idx = temp_ckpt_data.get('tag_to_idx')
        resumed_config_dict = temp_ckpt_data.get('config_dict')

        if not resumed_tags_list or not resumed_config_dict:
            logger.error("错误: 继续训练的检查点中缺少词汇表或配置信息"); return

        current_tags_list = resumed_tags_list
        current_tag_to_idx = resumed_tag_to_idx
        current_run_config = BaseConfig.from_dict(resumed_config_dict)
        current_run_config.DEVICE = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"已从检查点加载配置，NUM_CLASSES 将为 {len(current_tags_list)}")

        train_loader, _, _ = get_dataloader(current_run_config, mode="train", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx, rank=0, world_size=1)
        val_loader, _, _ = get_dataloader(current_run_config, mode="val", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx, rank=0, world_size=1)

        current_run_config.PRETRAINED = False
        model = get_model(current_run_config)
        model.to(current_run_config.DEVICE) # Move model to device *before* optimizer init

        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        else:
            logger.error(f"不支持的优化器: {current_run_config.OPTIMIZER}"); return


        if current_run_config.LR_SCHEDULER == "CosineAnnealingLR":
            t_max_for_resume = current_run_config.LR_SCHEDULER_PARAMS.get('T_max', current_run_config.EPOCHS)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_for_resume,
                                                             eta_min=current_run_config.LR_SCHEDULER_PARAMS.get('eta_min', 1e-6))
        elif current_run_config.LR_SCHEDULER == "StepLR":
             scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=current_run_config.LR_SCHEDULER_PARAMS.get('step_size', 10),
                                                  gamma=current_run_config.LR_SCHEDULER_PARAMS.get('gamma', 0.1))
        elif current_run_config.LR_SCHEDULER == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                             factor=current_run_config.LR_SCHEDULER_PARAMS.get('factor', 0.1),
                                                             patience=current_run_config.LR_SCHEDULER_PARAMS.get('patience', 5),
                                                             verbose=True)


        start_epoch, _, _, _ = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, current_run_config.DEVICE,
            require_optimizer_scheduler=True
        )
        logger.info(f"继续训练模式：模型、优化器、调度器状态已加载将从 Epoch {start_epoch} 开始")

    elif args.finetune_checkpoint:
        logger.info(f"准备从检查点进行微调: {args.finetune_checkpoint}")
        if not os.path.isfile(args.finetune_checkpoint):
            logger.error(f"错误: 用于微调的检查点文件未找到于 {args.finetune_checkpoint}"); return

        ckpt_data = torch.load(args.finetune_checkpoint, map_location='cpu')
        old_ckpt_model_state_dict = ckpt_data.get('model_state_dict')
        old_ckpt_config_dict = ckpt_data.get('config_dict')
        old_ckpt_tags_list = ckpt_data.get('tags_list')
        old_ckpt_tag_to_idx = ckpt_data.get('tag_to_idx')

        if not old_ckpt_model_state_dict or not old_ckpt_config_dict or not old_ckpt_tags_list:
            logger.error("错误: 微调检查点中缺少模型权重、配置或词汇表信息"); return

        current_run_config.PRETRAINED = False
        model = get_model(
            current_run_config,
            old_checkpoint_config_dict=old_ckpt_config_dict,
            old_checkpoint_state_dict=old_ckpt_model_state_dict,
            current_tags_list=current_tags_list, current_tag_to_idx=current_tag_to_idx,
            old_tags_list_from_ckpt=old_ckpt_tags_list, old_tag_to_idx_from_ckpt=old_ckpt_tag_to_idx
        )
        start_epoch = 0
        logger.info(f"微调模式：模型已创建/适配，优化器和调度器将重新初始化")

    else:
        logger.info("从头开始训练 (或使用 ImageNet 预训练权重)")
        model = get_model(current_run_config)
        start_epoch = 0

    model.to(current_run_config.DEVICE) # Ensure model is on the correct device

    if optimizer is None:
        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        else:
            logger.error(f"不支持的优化器: {current_run_config.OPTIMIZER}"); return
        logger.info(f"优化器 {current_run_config.OPTIMIZER} 已初始化，学习率: {current_run_config.LEARNING_RATE}")

    if scheduler is None:
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
                                                             mode='max',
                                                             factor=current_run_config.LR_SCHEDULER_PARAMS.get('factor', 0.1),
                                                             patience=current_run_config.LR_SCHEDULER_PARAMS.get('patience', 5),
                                                             verbose=True)
        if scheduler:
            logger.info(f"学习率调度器 {current_run_config.LR_SCHEDULER} 已初始化")
        else:
            logger.info("未使用学习率调度器")


    if current_run_config.LOSS_FN == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "AsymmetricLoss":
        criterion = AsymmetricLoss(**current_run_config.LOSS_FN_PARAMS).to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "WeightedBCEWithLogitsLoss":
        pos_weight_tensor = None
        if "pos_weight" in current_run_config.LOSS_FN_PARAMS:
            pos_weight_value = current_run_config.LOSS_FN_PARAMS["pos_weight"]
            if isinstance(pos_weight_value, (list, tuple)): # Ensure it's a list or tuple
                pos_weight_tensor = torch.tensor(pos_weight_value, device=current_run_config.DEVICE)
            else:
                logger.warning("WeightedBCEWithLogitsLoss 的 pos_weight 参数应为列表或元组。将不使用pos_weight。")
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(current_run_config.DEVICE)
    else:
        logger.error(f"不支持的损失函数: {current_run_config.LOSS_FN}"); return
    logger.info(f"损失函数: {current_run_config.LOSS_FN}")

    scaler = None
    if 'cuda' in str(current_run_config.DEVICE) and torch.cuda.is_available(): # Check if device string contains 'cuda'
        try:
            # Use torch.amp.GradScaler directly as torch.cuda.amp is deprecated
            scaler = torch.amp.GradScaler()
            logger.info("已启用混合精度训练 (GradScaler)")
        except AttributeError:
            logger.error("torch.amp.GradScaler 不可用 (可能 PyTorch 版本过旧)，混合精度训练将不可用")
            scaler = None
    elif 'cuda' in str(current_run_config.DEVICE) and not torch.cuda.is_available():
        logger.warning("配置设备为 CUDA，但 CUDA 不可用，将使用 CPU，混合精度已禁用")
        current_run_config.DEVICE = "cpu"
        model.to("cpu")
    else: # CPU 模式
        logger.info("使用 CPU 进行训练，混合精度已禁用")


    logger.info(f"开始训练，从 Epoch {start_epoch} 到 {current_run_config.EPOCHS -1}")
    logger.info(f"模型: {current_run_config.MODEL_NAME}, 目标类别数: {current_run_config.NUM_CLASSES}")
    logger.info(f"批次大小: {current_run_config.BATCH_SIZE}, 梯度累积步数: {current_run_config.ACCUMULATION_STEPS}")
    effective_batch_size = current_run_config.BATCH_SIZE * current_run_config.ACCUMULATION_STEPS # Single card
    logger.info(f"总有效批次大小: {effective_batch_size}")


    for epoch in range(start_epoch, current_run_config.EPOCHS):
        logger.info(f"Epoch {epoch}/{current_run_config.EPOCHS - 1}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, current_run_config, epoch, logger)

        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, current_run_config, epoch, logger)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metric)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch} 完成，当前学习率: {current_lr:.7f}")

        is_best = val_metric > best_metric_val
        if is_best:
            best_metric_val = val_metric
            logger.info(f"在 Epoch {epoch} 发现新的最佳验证指标 (F1-Micro): {best_metric_val:.4f}")

        checkpoint_name_prefix = args.run_name

        if current_run_config.SAVE_BEST_ONLY:
            if is_best:
                save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                current_tags_list, current_tag_to_idx, rank=0, # rank=0 for single card
                                filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True,
                                output_dir=current_run_config.OUTPUT_DIR)
        else:
            save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                            current_tags_list, current_tag_to_idx, rank=0,
                            filename_prefix=f"{checkpoint_name_prefix}_epoch_{epoch}", is_best_marker_for_log=False,
                            output_dir=current_run_config.OUTPUT_DIR)
            if is_best:
                save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                current_tags_list, current_tag_to_idx, rank=0,
                                filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True,
                                output_dir=current_run_config.OUTPUT_DIR)

        save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                        current_tags_list, current_tag_to_idx, rank=0,
                        filename_prefix=f"{checkpoint_name_prefix}_last", is_best_marker_for_log=False,
                        output_dir=current_run_config.OUTPUT_DIR)

    logger.info("训练完成")
    logger.info(f"最佳验证指标 (F1-Micro): {best_metric_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 图像多标签分类训练脚本 (单卡)") # Updated description
    parser.add_argument("--config_name", type=str, default="eva02_L_clip336_merged2b_config.py",
                        help="配置文件名 (例如: example_swin_config.py)，位于 configs/ 目录下")
    parser.add_argument("--config_class_name", type=str, default="Eva02LargeClip336Merged2BConfig",
                        help="配置文件中配置类的名称 (例如: ExampleSwinConfig)")
    parser.add_argument("--run_name", type=str, default="danbooru_experiment_single_card", # Updated default run_name
                        help="本次运行的名称，用于日志和模型保存")

    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="从指定的检查点路径继续训练 (加载模型、优化器、调度器状态)")
    parser.add_argument("--finetune_checkpoint", type=str, default=None,
                        help="从指定的检查点路径进行微调 (加载模型 backbone 权重，可适配新标签集，重新初始化优化器/调度器)")

    parser.add_argument("--device", type=str, default=None, # choices=['cuda', 'cpu', 'cuda:0', 'cuda:1' etc.]
                        help="指定运行设备 (例如 'cuda:0', 'cpu')。如果未指定，则使用配置文件中的 DEVICE 或自动检测。")


    args = parser.parse_args()

    # 使用 basicConfig 进行简单的日志设置，因为 setup_logger 是为 DDP rank 设计的
    # 如果需要文件日志，可以在 main 函数开始时使用 setup_logger(rank=0)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


    if args.resume_checkpoint and args.finetune_checkpoint:
        logging.error("错误: --resume_checkpoint 和 --finetune_checkpoint 参数不能同时指定")
    else:
        main(args)
