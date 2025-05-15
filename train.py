import argparse
import os
import platform # 用于检测操作系统
import tempfile # 用于创建临时文件
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import importlib
import logging
from contextlib import nullcontext

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from src.dataset import get_dataloader
from src.model import get_model
from src.utils import AverageMeter, save_checkpoint, load_checkpoint, setup_logger, calculate_metrics
from src.losses import AsymmetricLoss, WeightedBCEWithLogitsLoss
from configs.base_config import BaseConfig

logger = None

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, current_epoch_config, current_epoch_num, global_logger, rank, world_size):
    """训练一个 epoch"""
    model.train()
    loss_meter = AverageMeter()

    if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch') and isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(current_epoch_num)

    optimizer.zero_grad()

    sdp_context = nullcontext()
    if current_epoch_config.ENABLE_SDP_ATTENTION and hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0":
        try:
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
            if current_epoch_num == 0 and rank == 0:
                 global_logger.info("已尝试启用 PyTorch SDPA via torch.nn.attention.sdpa_kernel")
        except Exception as e:
            if current_epoch_num == 0 and rank == 0:
                global_logger.warning(f"尝试启用 SDPA (torch.nn.attention.sdpa_kernel) 时出错: {e}，将使用标准注意力实现")

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num} [训练]", leave=False, disable=(rank != 0))
    with sdp_context:
        for i, (images, targets) in enumerate(progress_bar):
            images, targets = images.to(current_epoch_config.DEVICE), targets.to(current_epoch_config.DEVICE)

            autocast_device_type = 'cuda' if isinstance(current_epoch_config.DEVICE, int) else 'cpu'

            with torch.amp.autocast(device_type=autocast_device_type, enabled=(scaler is not None and autocast_device_type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss = loss / current_epoch_config.ACCUMULATION_STEPS

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % current_epoch_config.ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            loss_meter.update(loss.item() * current_epoch_config.ACCUMULATION_STEPS, images.size(0))
            if rank == 0:
                progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    if rank == 0:
        global_logger.info(f"Epoch {current_epoch_num} [训练] 平均损失: {loss_meter.avg:.4f}")
    return loss_meter.avg


def validate_one_epoch(model, dataloader, criterion, current_epoch_config, current_epoch_num, global_logger, rank, world_size):
    """验证一个 epoch"""
    model.eval()
    loss_meter = AverageMeter()

    local_all_preds_sigmoid_list = []
    local_all_targets_list = []

    sdp_context = nullcontext()
    if current_epoch_config.ENABLE_SDP_ATTENTION and hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0":
        try:
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
        except Exception:
            pass

    progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch_num} [验证]", leave=False, disable=(rank != 0))
    
    # MODIFICATION START: Determine device type for autocast
    autocast_device_type = 'cuda' if isinstance(current_epoch_config.DEVICE, int) or \
                                   (isinstance(current_epoch_config.DEVICE, str) and 'cuda' in current_epoch_config.DEVICE) \
                                else 'cpu'
    # MODIFICATION END

    with sdp_context:
        with torch.no_grad():
            # MODIFICATION START: Add autocast context for validation
            with torch.amp.autocast(device_type=autocast_device_type, enabled=(autocast_device_type == 'cuda')):
            # MODIFICATION END
                for images, targets in progress_bar:
                    images, targets_gpu = images.to(current_epoch_config.DEVICE), targets.to(current_epoch_config.DEVICE)

                    outputs = model(images)
                    loss = criterion(outputs, targets_gpu)
                    loss_meter.update(loss.item(), images.size(0))

                    preds_sigmoid = torch.sigmoid(outputs)
                    local_all_preds_sigmoid_list.append(preds_sigmoid.cpu())
                    local_all_targets_list.append(targets.cpu())

                    if rank == 0:
                        progress_bar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    if not local_all_preds_sigmoid_list:
        if rank == 0 and world_size > 1:
             global_logger.warning(f"Rank {rank} 在验证时没有收集到任何预测/目标。")

        num_classes = current_epoch_config.NUM_CLASSES
        if num_classes <=0:
            if rank == 0: global_logger.error("NUM_CLASSES 未正确设置，无法创建占位符张量。")
            return loss_meter.avg, 0.0

        local_preds_concat = torch.empty((0, num_classes), dtype=torch.float32)
        local_targets_concat = torch.empty((0, num_classes), dtype=torch.float32)
    else:
        local_preds_concat = torch.cat(local_all_preds_sigmoid_list)
        local_targets_concat = torch.cat(local_all_targets_list)

    all_preds_sigmoid_gathered = None
    all_targets_gathered = None

    if world_size > 1:
        gathered_preds_objects = [None for _ in range(world_size)]
        gathered_targets_objects = [None for _ in range(world_size)]

        if dist.is_initialized():
            dist.barrier()

        dist.all_gather_object(gathered_preds_objects, local_preds_concat)
        dist.all_gather_object(gathered_targets_objects, local_targets_concat)

        if rank == 0:
            valid_preds = [p for p in gathered_preds_objects if p is not None and p.shape[0] > 0]
            valid_targets = [t for t in gathered_targets_objects if t is not None and t.shape[0] > 0]

            if valid_preds:
                all_preds_sigmoid_gathered = torch.cat(valid_preds)
            else:
                all_preds_sigmoid_gathered = torch.empty((0, current_epoch_config.NUM_CLASSES), dtype=torch.float32)

            if valid_targets:
                all_targets_gathered = torch.cat(valid_targets)
            else:
                all_targets_gathered = torch.empty((0, current_epoch_config.NUM_CLASSES), dtype=torch.float32)
    else:
        all_preds_sigmoid_gathered = local_preds_concat
        all_targets_gathered = local_targets_concat

    current_val_metric_f1_micro = 0.0

    if rank == 0:
        if hasattr(dataloader, 'dataset') and dataloader.dataset is not None:
            original_val_dataset_len = len(dataloader.dataset)
            if all_preds_sigmoid_gathered is not None and all_preds_sigmoid_gathered.shape[0] > original_val_dataset_len:
                all_preds_sigmoid_gathered = all_preds_sigmoid_gathered[:original_val_dataset_len]
            if all_targets_gathered is not None and all_targets_gathered.shape[0] > original_val_dataset_len:
                all_targets_gathered = all_targets_gathered[:original_val_dataset_len]
        else:
            global_logger.warning("无法确定原始验证数据集长度以进行潜在的指标计算截断。")

        if all_preds_sigmoid_gathered is not None and all_targets_gathered is not None and \
           all_preds_sigmoid_gathered.shape[0] > 0 and \
           all_preds_sigmoid_gathered.shape[0] == all_targets_gathered.shape[0]: # 确保预测和目标数量一致

            metrics = calculate_metrics(all_preds_sigmoid_gathered, all_targets_gathered, threshold=current_epoch_config.INFERENCE_THRESHOLD)

            if metrics['f1_micro'] == 0 and metrics['f1_macro'] == 0 and metrics['exact_match_ratio'] == 0:
                global_logger.info(f"Epoch {current_epoch_num} [验证] 使用默认阈值 {current_epoch_config.INFERENCE_THRESHOLD} 的所有指标为零，尝试其他阈值...")

            global_logger.info(f"Epoch {current_epoch_num} [验证] 平均损失: {loss_meter.avg:.4f}")
            global_logger.info(f"Epoch {current_epoch_num} [验证] F1-Micro: {metrics['f1_micro']:.4f}, F1-Macro: {metrics['f1_macro']:.4f}, mAP-Macro: {metrics['mAP_macro']:.4f}, ExactMatch: {metrics['exact_match_ratio']:.4f}")

            current_val_metric_f1_micro = metrics['f1_micro']
        elif rank == 0: # 只有rank 0打印这个错误
            global_logger.error(f"Epoch {current_epoch_num} [验证] 无法收集到足够的或匹配的预测/目标来进行指标计算。Preds shape: {all_preds_sigmoid_gathered.shape if all_preds_sigmoid_gathered is not None else 'None'}, Targets shape: {all_targets_gathered.shape if all_targets_gathered is not None else 'None'}")


    return loss_meter.avg, current_val_metric_f1_micro


def main_worker(rank, world_size, args, current_run_config_class_ref, using_gpu, temp_file_for_gloo_cleanup_path):
    """每个 DDP 进程的工作函数"""
    global logger

    current_dist_url = args.dist_url
    backend_to_use = args.dist_backend

    if not current_dist_url.startswith("file://"):
        os.environ['MASTER_ADDR'] = current_dist_url.split(':')[0].replace('tcp://', '')
        os.environ['MASTER_PORT'] = current_dist_url.split(':')[1]

    if rank == 0:
        logging.info(f"进程 {rank}: 正在使用后端 '{backend_to_use}' 和初始化方法 '{current_dist_url}' 初始化进程组...")

    try:
        dist.init_process_group(backend=backend_to_use, init_method=current_dist_url, world_size=world_size, rank=rank)
    except RuntimeError as e:
        if rank == 0:
            logging.error(f"进程 {rank}: 使用后端 '{backend_to_use}' 和方法 '{current_dist_url}' 初始化进程组失败: {e}")

        if backend_to_use == 'nccl' and platform.system() == "Linux" and using_gpu:
            if rank == 0: logging.warning(f"进程 {rank}: NCCL 后端失败，正在尝试回退到 GLOO 后端 (使用FileStore)...")
            backend_to_use = 'gloo'
            current_dist_url = args.gloo_file_uri
            if rank == 0: logging.info(f"进程 {rank}: NCCL回退：更新 dist_url 为 GLOO FileStore: {current_dist_url}")

            if not current_dist_url:
                if rank == 0: logging.error(f"进程 {rank}: NCCL回退到GLOO失败：未提供FileStore的有效URI (args.gloo_file_uri为空)。")
                return

            try:
                dist.init_process_group(backend=backend_to_use, init_method=current_dist_url, world_size=world_size, rank=rank)
                if rank == 0: logging.info(f"进程 {rank}: 成功使用 GLOO 后端 (FileStore) 进行初始化 (NCCL 回退，使用 {current_dist_url})。")
            except RuntimeError as e_gloo:
                if rank == 0: logging.error(f"进程 {rank}: GLOO 后端 (FileStore) 回退也失败: {e_gloo}")
                return
        else:
            return

    if using_gpu:
        torch.cuda.set_device(rank)

    current_run_config = current_run_config_class_ref()
    current_run_config.DEVICE = rank if using_gpu else 'cpu'


    logger = setup_logger(current_run_config.LOG_DIR, args.run_name, rank=rank)
    if rank == 0:
        logger.info(f"使用配置: {args.config_name} (类: {args.config_class_name})")
        logger.info(f"运行名称: {args.run_name}")
        logger.info(f"分布式训练: world_size={world_size}, 使用的设备: {'GPUs ' + args.gpu_ids if using_gpu else 'CPUs'}, 后端: {backend_to_use}")
        if current_run_config.ENABLE_SDP_ATTENTION and torch.__version__ < "2.0.0":
            logger.warning("配置中 ENABLE_SDP_ATTENTION=True, 但 PyTorch 版本低于 2.0.0，SDPA 可能不可用")

    if rank == 0: logger.info("正在初始化数据加载器和词汇表...")
    train_loader, current_tags_list, current_tag_to_idx = get_dataloader(
        current_run_config, mode="train", rank=rank, world_size=world_size
    )
    if rank == 0: logger.info(f"目标词汇表大小 (NUM_CLASSES): {current_run_config.NUM_CLASSES} 个标签")

    val_loader, _, _ = get_dataloader(
        current_run_config, mode="val", tags_list=current_tags_list,
        tag_to_idx=current_tag_to_idx, rank=rank, world_size=world_size
    )

    model = None
    optimizer = None
    scheduler = None
    start_epoch = 0
    best_metric_val = 0.0

    if args.resume_checkpoint:
        if rank == 0: logger.info(f"准备从检查点继续训练: {args.resume_checkpoint}")
        if not os.path.isfile(args.resume_checkpoint):
            if rank == 0: logger.error(f"错误: 继续训练的检查点文件未找到: {args.resume_checkpoint}")
            return

        map_loc = f'cuda:{rank}' if using_gpu else 'cpu'
        temp_ckpt_data = torch.load(args.resume_checkpoint, map_location=map_loc)

        resumed_tags_list = temp_ckpt_data.get('tags_list')
        resumed_tag_to_idx = temp_ckpt_data.get('tag_to_idx')
        resumed_config_dict = temp_ckpt_data.get('config_dict')

        if not resumed_tags_list or not resumed_config_dict:
            if rank == 0: logger.error("错误: 继续训练的检查点中缺少词汇表或配置信息")
            return

        current_tags_list = resumed_tags_list
        current_tag_to_idx = resumed_tag_to_idx
        current_run_config = BaseConfig.from_dict(resumed_config_dict)
        current_run_config.DEVICE = rank if using_gpu else 'cpu'
        if rank == 0: logger.info(f"已从检查点加载配置，NUM_CLASSES 为 {len(current_tags_list)}")

        train_loader, _, _ = get_dataloader(current_run_config, mode="train", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx, rank=rank, world_size=world_size)
        val_loader, _, _ = get_dataloader(current_run_config, mode="val", tags_list=current_tags_list, tag_to_idx=current_tag_to_idx, rank=rank, world_size=world_size)

        current_run_config.PRETRAINED = False
        model = get_model(current_run_config)
        model = model.to(current_run_config.DEVICE)

        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        else:
            if rank == 0: logger.error(f"不支持的优化器: {current_run_config.OPTIMIZER}")
            return

        if current_run_config.LR_SCHEDULER == "CosineAnnealingLR":
            t_max_for_resume = current_run_config.LR_SCHEDULER_PARAMS.get('T_max', current_run_config.EPOCHS)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_for_resume, eta_min=current_run_config.LR_SCHEDULER_PARAMS.get('eta_min', 1e-6))
        elif current_run_config.LR_SCHEDULER == "StepLR":
             scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=current_run_config.LR_SCHEDULER_PARAMS.get('step_size', 10),
                                                  gamma=current_run_config.LR_SCHEDULER_PARAMS.get('gamma', 0.1))
        elif current_run_config.LR_SCHEDULER == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                             factor=current_run_config.LR_SCHEDULER_PARAMS.get('factor', 0.1),
                                                             patience=current_run_config.LR_SCHEDULER_PARAMS.get('patience', 5),
                                                             verbose= (rank==0 and world_size==1) )

        if world_size > 1:
            if using_gpu:
                model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            else:
                model = DDP(model, find_unused_parameters=True)

        start_epoch, _, _, _ = load_checkpoint(
            args.resume_checkpoint, model, optimizer, scheduler, device=current_run_config.DEVICE,
            require_optimizer_scheduler=True
        )
        if rank == 0: logger.info(f"继续训练模式：模型、优化器、调度器状态已加载，将从 Epoch {start_epoch} 开始")

    elif args.finetune_checkpoint:
        if rank == 0: logger.info(f"准备从检查点进行微调: {args.finetune_checkpoint}")
        if not os.path.isfile(args.finetune_checkpoint):
            if rank == 0: logger.error(f"错误: 用于微调的检查点文件未找到: {args.finetune_checkpoint}")
            return

        map_location = 'cpu'
        ckpt_data = torch.load(args.finetune_checkpoint, map_location=map_location)
        old_ckpt_model_state_dict = ckpt_data.get('model_state_dict')
        old_ckpt_config_dict = ckpt_data.get('config_dict')
        old_ckpt_tags_list = ckpt_data.get('tags_list')
        old_ckpt_tag_to_idx = ckpt_data.get('tag_to_idx')

        if not old_ckpt_model_state_dict or not old_ckpt_config_dict or not old_ckpt_tags_list:
            if rank == 0: logger.error("错误: 微调检查点中缺少模型权重、配置或词汇表信息")
            return

        current_run_config.PRETRAINED = False
        model = get_model(
            current_run_config,
            old_checkpoint_config_dict=old_ckpt_config_dict,
            old_checkpoint_state_dict=old_ckpt_model_state_dict,
            current_tags_list=current_tags_list, current_tag_to_idx=current_tag_to_idx,
            old_tags_list_from_ckpt=old_ckpt_tags_list, old_tag_to_idx_from_ckpt=old_ckpt_tag_to_idx
        )
        model = model.to(current_run_config.DEVICE)
        start_epoch = 0
        if rank == 0: logger.info(f"微调模式：模型已创建/适配，优化器和调度器将重新初始化")

    else:
        if rank == 0: logger.info("从头开始训练 (或使用 ImageNet 预训练权重)")
        model = get_model(current_run_config)
        model = model.to(current_run_config.DEVICE)
        start_epoch = 0

    if not args.resume_checkpoint:
        if world_size > 1:
            if using_gpu:
                model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
            else:
                 model = DDP(model, find_unused_parameters=True)

    if optimizer is None:
        if current_run_config.OPTIMIZER == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=current_run_config.LEARNING_RATE, weight_decay=current_run_config.WEIGHT_DECAY)
        elif current_run_config.OPTIMIZER == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=current_run_config.LEARNING_RATE, momentum=0.9, weight_decay=current_run_config.WEIGHT_DECAY)
        else:
            if rank == 0: logger.error(f"不支持的优化器: {current_run_config.OPTIMIZER}")
            return
        if rank == 0: logger.info(f"优化器 {current_run_config.OPTIMIZER} 已初始化，学习率: {current_run_config.LEARNING_RATE}")

    if scheduler is None:
        if current_run_config.LR_SCHEDULER == "CosineAnnealingLR":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=current_run_config.LR_SCHEDULER_PARAMS.get('T_max', current_run_config.EPOCHS), eta_min=current_run_config.LR_SCHEDULER_PARAMS.get('eta_min', 1e-6))
        elif current_run_config.LR_SCHEDULER == "StepLR":
             scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=current_run_config.LR_SCHEDULER_PARAMS.get('step_size', 10),
                                                  gamma=current_run_config.LR_SCHEDULER_PARAMS.get('gamma', 0.1))
        elif current_run_config.LR_SCHEDULER == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                             factor=current_run_config.LR_SCHEDULER_PARAMS.get('factor', 0.1),
                                                             patience=current_run_config.LR_SCHEDULER_PARAMS.get('patience', 5),
                                                             verbose=(rank==0 and world_size==1) )
        if scheduler and rank == 0: logger.info(f"学习率调度器 {current_run_config.LR_SCHEDULER} 已初始化")
        elif not scheduler and rank == 0: logger.info("未使用学习率调度器")

    if current_run_config.LOSS_FN == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "AsymmetricLoss":
        criterion = AsymmetricLoss(**current_run_config.LOSS_FN_PARAMS).to(current_run_config.DEVICE)
    elif current_run_config.LOSS_FN == "WeightedBCEWithLogitsLoss":
        pos_weight_tensor = None
        if "pos_weight" in current_run_config.LOSS_FN_PARAMS:
            pos_weight_value = current_run_config.LOSS_FN_PARAMS["pos_weight"]
            if isinstance(pos_weight_value, (list, tuple)):
                pos_weight_tensor = torch.tensor(pos_weight_value, device=current_run_config.DEVICE)
            else:
                if rank == 0: logger.warning("WeightedBCEWithLogitsLoss 的 pos_weight 参数应为列表或元组。将不使用pos_weight。")
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(current_run_config.DEVICE)
    else:
        if rank == 0: logger.error(f"不支持的损失函数: {current_run_config.LOSS_FN}")
        return
    if rank == 0: logger.info(f"损失函数: {current_run_config.LOSS_FN}")

    scaler = None
    if using_gpu:
        scaler = torch.amp.GradScaler()
        if rank == 0: logger.info("已启用混合精度训练 (GradScaler)")

    if rank == 0:
        logger.info(f"开始训练，从 Epoch {start_epoch} 到 {current_run_config.EPOCHS -1}")
        logger.info(f"模型: {current_run_config.MODEL_NAME}, 目标类别数: {current_run_config.NUM_CLASSES}")
        logger.info(f"批次大小 (每个进程): {current_run_config.BATCH_SIZE}, 梯度累积步数: {current_run_config.ACCUMULATION_STEPS}")
        effective_batch_size = current_run_config.BATCH_SIZE * world_size * current_run_config.ACCUMULATION_STEPS
        logger.info(f"总有效批次大小: {effective_batch_size}")

    try:
        for epoch in range(start_epoch, current_run_config.EPOCHS):
            if rank == 0: logger.info(f"Epoch {epoch}/{current_run_config.EPOCHS - 1}")

            if world_size > 1 and hasattr(train_loader.sampler, 'set_epoch') and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            if world_size > 1 and hasattr(val_loader.sampler, 'set_epoch') and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, current_run_config, epoch, logger, rank, world_size)

            val_loss, val_metric_f1_micro_rank0 = validate_one_epoch(model, val_loader, criterion, current_run_config, epoch, logger, rank, world_size)

            synced_val_metric = val_metric_f1_micro_rank0
            if world_size > 1 and dist.is_initialized():
                metric_device_val_str = str(current_run_config.DEVICE) # 确保是字符串
                if not metric_device_val_str.startswith('cuda') and not metric_device_val_str.startswith('cpu'): # 如果只是rank数字
                     metric_device_val = f'cuda:{current_run_config.DEVICE}' if using_gpu else 'cpu'
                else:
                     metric_device_val = current_run_config.DEVICE

                metric_tensor = torch.tensor(val_metric_f1_micro_rank0, device=metric_device_val)
                dist.broadcast(metric_tensor, src=0)
                synced_val_metric = metric_tensor.item()

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(synced_val_metric)
                else:
                    scheduler.step()

            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch} 完成，当前学习率: {current_lr:.7f}")

                is_best = synced_val_metric > best_metric_val
                if is_best:
                    best_metric_val = synced_val_metric
                    logger.info(f"在 Epoch {epoch} 发现新的最佳验证指标 (F1-Micro): {best_metric_val:.4f}")

                checkpoint_name_prefix = args.run_name

                if current_run_config.SAVE_BEST_ONLY:
                    if is_best:
                        save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                        current_tags_list, current_tag_to_idx, rank,
                                        filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True,
                                        output_dir=current_run_config.OUTPUT_DIR)
                else:
                    save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                    current_tags_list, current_tag_to_idx, rank,
                                    filename_prefix=f"{checkpoint_name_prefix}_epoch_{epoch}", is_best_marker_for_log=False,
                                    output_dir=current_run_config.OUTPUT_DIR)
                    if is_best:
                         save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                        current_tags_list, current_tag_to_idx, rank,
                                        filename_prefix=checkpoint_name_prefix, is_best_marker_for_log=True,
                                        output_dir=current_run_config.OUTPUT_DIR)

                save_checkpoint(epoch, model, optimizer, scheduler, current_run_config,
                                current_tags_list, current_tag_to_idx, rank,
                                filename_prefix=f"{checkpoint_name_prefix}_last", is_best_marker_for_log=False,
                                output_dir=current_run_config.OUTPUT_DIR)

            if world_size > 1 and dist.is_initialized():
                dist.barrier()

        if rank == 0:
            logger.info("训练完成")
            logger.info(f"最佳验证指标 (F1-Micro): {best_metric_val:.4f}")

    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        if rank == 0 and temp_file_for_gloo_cleanup_path and os.path.exists(temp_file_for_gloo_cleanup_path):
            try:
                os.remove(temp_file_for_gloo_cleanup_path)
                logging.info(f"进程 {rank} 已清理临时文件: {temp_file_for_gloo_cleanup_path}")
            except OSError as e:
                logging.warning(f"进程 {rank} 清理临时文件 {temp_file_for_gloo_cleanup_path} 失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 图像多标签分类训练脚本 (多卡DDP)")
    parser.add_argument("--config_name", type=str, default="eva02_L_clip336_merged2b_config.py",
                        help="配置文件名")
    parser.add_argument("--config_class_name", type=str, default="Eva02LargeClip336Merged2BConfig",
                        help="配置文件中配置类的名称")
    parser.add_argument("--run_name", type=str, default="danbooru_experiment_ddp",
                        help="本次运行的名称")

    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="从指定的检查点路径继续训练")
    parser.add_argument("--finetune_checkpoint", type=str, default=None,
                        help="从指定的检查点路径进行微调")

    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="要使用的 GPU ID 列表 (例如 '0,1,2')。为空则使用CPU。")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:23456", type=str,
                        help="用于分布式训练的 URL (tcp://...) 或 FileStore 路径 (file:///...)")

    # 内部参数，不暴露给用户，由脚本自动设置
    parser.add_argument("--dist_backend", type=str, default="gloo", help=argparse.SUPPRESS)
    parser.add_argument("--gloo_file_uri", type=str, default=None, help=argparse.SUPPRESS) # 用于NCCL失败时回退到GLOO FileStore
    parser.add_argument("--temp_file_for_gloo_cleanup_path", type=str, default=None, help=argparse.SUPPRESS) # 用于清理


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if args.resume_checkpoint and args.finetune_checkpoint:
        logging.error("错误: --resume_checkpoint 和 --finetune_checkpoint 参数不能同时指定")
    else:
        world_size = 1
        using_gpu = False
        gloo_temp_file_main_path = None # 用于主进程创建并传递给子进程进行清理

        if torch.cuda.is_available() and args.gpu_ids and args.gpu_ids.strip():
            try:
                parsed_gpu_ids_str = [s.strip() for s in args.gpu_ids.split(',') if s.strip().isdigit()]
                if not parsed_gpu_ids_str:
                    logging.warning(f"--gpu_ids '{args.gpu_ids}' 未包含有效的数字ID。")
                else:
                    num_available_gpus = torch.cuda.device_count()
                    valid_gpu_ids_for_run = []
                    for gpu_id_str in parsed_gpu_ids_str:
                        gpu_id_int = int(gpu_id_str)
                        if 0 <= gpu_id_int < num_available_gpus:
                            valid_gpu_ids_for_run.append(str(gpu_id_int))
                        else:
                            logging.warning(f"请求的 GPU ID {gpu_id_int} 超出可用范围 (0-{num_available_gpus-1})，将被忽略。")

                    if not valid_gpu_ids_for_run:
                        logging.warning(f"所有请求的 GPU ID 均无效或超出范围。")
                    else:
                        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(valid_gpu_ids_for_run)
                        world_size = len(valid_gpu_ids_for_run)
                        using_gpu = True
                        args.gpu_ids = ",".join(valid_gpu_ids_for_run)
                        logging.info(f"设置 CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}', world_size={world_size} (GPU训练)")
            except Exception as e:
                logging.error(f"解析 --gpu_ids ('{args.gpu_ids}') 时出错: {e}.")

        # 决定后端和初始化方法
        # 1. 总是准备 GLOO FileStore 的路径和 URI，以备 NCCL 失败时回退，或非 Linux GPU 环境直接使用
        try:
            with tempfile.NamedTemporaryFile(delete=False, prefix="dist_sync_", suffix=".tmp") as tmpfile:
                gloo_temp_file_main_path = tmpfile.name
            abs_path = os.path.abspath(gloo_temp_file_main_path)
            uri_path = abs_path.replace(os.sep, '/')
            if platform.system() == "Windows":
                args.gloo_file_uri = f"file:///{uri_path}"
            else:
                args.gloo_file_uri = f"file://{uri_path}"
            args.temp_file_for_gloo_cleanup_path = gloo_temp_file_main_path
            logging.info(f"已准备 GLOO FileStore URI (用于备用或GLOO主选): {args.gloo_file_uri} (临时文件: {gloo_temp_file_main_path})")
        except Exception as e_tmp:
            logging.error(f"创建 GLOO FileStore 临时文件失败: {e_tmp}. 如果 GLOO 被使用，将尝试 TCP URL。")
            args.gloo_file_uri = None
            args.temp_file_for_gloo_cleanup_path = None


        # 2. 根据平台和 GPU 使用情况选择首选后端和 dist_url
        if platform.system() == "Linux" and using_gpu:
            args.dist_backend = 'nccl'
            # args.dist_url 保持为用户提供的TCP URL (或默认TCP) 用于 NCCL
            logging.info(f"Linux GPU 环境，首选后端: nccl, 主 init_method (TCP): {args.dist_url}.")
            # args.gloo_file_uri 已准备好作为 NCCL 失败时的备用 FileStore URI
        else: # Windows, macOS, 或 CPU
            args.dist_backend = 'gloo'
            if args.gloo_file_uri: # 如果 FileStore URI 准备好了
                args.dist_url = args.gloo_file_uri # GLOO 主要使用 FileStore
                logging.info(f"非Linux GPU或CPU环境，首选后端: gloo, 主 init_method (FileStore): {args.dist_url}")
            else: # FileStore 准备失败，GLOO 回退到TCP
                logging.warning(f"非Linux GPU或CPU环境，后端: gloo。FileStore准备失败，将使用TCP init_method: {args.dist_url}")
                # args.dist_url 保持为用户提供的TCP URL (或默认TCP)

        if not using_gpu: # 如果最终确定不使用GPU（例如，CUDA不可用或未指定有效GPU ID）
            world_size = 1
            args.gpu_ids = ""
            args.dist_backend = 'gloo' # CPU模式也用gloo
            if args.gloo_file_uri: # CPU单进程也优先用FileStore如果准备好了
                args.dist_url = args.gloo_file_uri
            # 即使是单CPU进程，也设置一个dist_url，尽管DDP在这种情况下不会进行实际通信
            logging.info(f"CUDA 不可用或未指定有效GPU。将使用CPU进行单进程训练 (world_size={world_size}, backend={args.dist_backend}, init_method={args.dist_url})。")

        try:
            config_module_name = args.config_name.replace('.py', '')
            current_config_module = importlib.import_module(f"configs.{config_module_name}")
            CurrentConfigClass = getattr(current_config_module, args.config_class_name)
        except ModuleNotFoundError:
            logging.error(f"错误: 配置文件模块 'configs.{config_module_name}' 未找到")
            if gloo_temp_file_main_path and os.path.exists(gloo_temp_file_main_path): os.remove(gloo_temp_file_main_path)
            exit(1)
        except AttributeError:
            logging.error(f"错误: 在配置文件模块 'configs.{config_module_name}' 中未找到配置类 '{args.config_class_name}'")
            if gloo_temp_file_main_path and os.path.exists(gloo_temp_file_main_path): os.remove(gloo_temp_file_main_path)
            exit(1)

        try:
            # 将 temp_file_for_gloo_cleanup_path 传递给 worker
            mp.spawn(main_worker, nprocs=world_size, args=(world_size, args, CurrentConfigClass, using_gpu, args.temp_file_for_gloo_cleanup_path))
        finally:
            # 主进程在这里最后再尝试清理一次，以防 worker 的 rank 0 没能正确清理或 worker 未成功启动
            if args.temp_file_for_gloo_cleanup_path and os.path.exists(args.temp_file_for_gloo_cleanup_path):
                try:
                    os.remove(args.temp_file_for_gloo_cleanup_path)
                    logging.info(f"主进程已清理临时文件: {args.temp_file_for_gloo_cleanup_path}")
                except OSError as e:
                    logging.warning(f"主进程清理临时文件 {args.temp_file_for_gloo_cleanup_path} 失败: {e}")
