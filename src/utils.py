import os
import torch
import logging
import shutil # 用于 shutil.copyfile

def setup_logger(log_dir, run_name, log_level=logging.INFO, rank=0):
    """配置日志记录器，同时输出到控制台和文件。只有 rank 0 会实际写入。"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger_instance = logging.getLogger(run_name + f"_rank{rank}") # 每个 rank 一个 logger 实例，避免冲突
    logger_instance.setLevel(log_level)
    
    # 清除已存在的 handlers，防止重复添加 (尤其在 mp.spawn 中)
    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    if rank == 0: # 只有主进程 (rank 0) 设置文件和控制台处理器
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_instance.addHandler(console_handler)
    else: # 其他进程可以不记录，或者记录到 NullHandler
        logger_instance.addHandler(logging.NullHandler())
            
    return logger_instance

class AverageMeter:
    """计算并存储平均值和当前值。"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0


def save_checkpoint(epoch, model, optimizer, scheduler, config_to_save, 
                    tags_list, tag_to_idx, rank, # 添加 rank 参数
                    filename_prefix="checkpoint", 
                    is_best_marker_for_log=False, 
                    output_dir="trained_models/"):
    """
    保存模型检查点。仅在 rank 0 执行。
    如果模型是 DDP 实例，则保存 model.module.state_dict()。
    """
    if rank != 0: # 只有主进程 (rank 0) 保存检查点
        return

    # 如果 model 是 DDP 实例，获取其 .module 属性以获取原始模型
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    state = {
        'epoch': epoch + 1, 
        'model_state_dict': model_to_save.state_dict(), # 保存原始模型的 state_dict
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'tags_list': tags_list,
        'tag_to_idx': tag_to_idx,
        'is_best_model_marker': is_best_marker_for_log 
    }
    
    if hasattr(config_to_save, 'to_dict') and callable(config_to_save.to_dict):
        state['config_dict'] = config_to_save.to_dict()
    elif isinstance(config_to_save, dict):
        state['config_dict'] = config_to_save
    else:
        try:
            state['config_dict'] = vars(config_to_save)
        except TypeError:
            logging.warning("无法将配置对象转换为字典进行保存。检查点中可能缺少配置信息。")
            state['config_dict'] = None

    target_filename = f"{filename_prefix}.pth.tar"
    filepath = os.path.join(output_dir, target_filename)
    
    try:
        torch.save(state, filepath)
        log_message = f"检查点已保存到: {filepath} (Epoch {epoch})"
        if is_best_marker_for_log:
            log_message += " [标记为当前最佳]"
        logging.info(log_message) # logging.info 应该已经被 rank 0 的 logger 处理
    except Exception as e:
        logging.error(f"保存检查点 {filepath} 时出错: {e}")


def load_checkpoint(checkpoint_path, model_instance, 
                    optimizer=None, scheduler=None, device="cuda", # device 是目标 rank/gpu
                    require_optimizer_scheduler=False): 
    if not os.path.isfile(checkpoint_path):
        logging.error(f"错误: 检查点文件未找到于 {checkpoint_path}")
        raise FileNotFoundError(f"检查点文件未找到于 {checkpoint_path}")
    
    # device 参数现在是目标加载设备 (例如，当前进程的 GPU rank)
    map_location = f'cuda:{device}' if isinstance(device, int) else device
    logging.info(f"正在从检查点加载: {checkpoint_path} 到 {map_location}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location) 
    
    # 如果 model_instance 是 DDP 实例，我们需要加载到其 .module
    model_to_load_on = model_instance.module if isinstance(model_instance, torch.nn.parallel.DistributedDataParallel) else model_instance

    model_state_dict = checkpoint['model_state_dict']
    # DDP 保存时已经是 module.state_dict()，所以加载时不需要额外处理 'module.' 前缀
    # 但如果加载的是非 DDP 保存的权重到 DDP 模型 (的 module)，或者反之，则需要处理
    # 为了通用性，保留移除 'module.' 的逻辑，以防加载非DDP模型到DDP的module，或加载DDP模型到非DDP模型
    if any(key.startswith('module.') for key in model_state_dict.keys()):
        # logging.info("检测到 'module.' 前缀，正在移除...") # 这个日志可能过于频繁
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    load_status = model_to_load_on.load_state_dict(model_state_dict, strict=False)
    if load_status.missing_keys:
        logging.warning(f"加载模型权重时发现缺失的键: {load_status.missing_keys}")
    if load_status.unexpected_keys:
        logging.warning(f"加载模型权重时发现意外的键: {load_status.unexpected_keys}")
    logging.info("模型权重加载完成。")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("优化器状态加载完成。")
        except Exception as e:
            logging.warning(f"加载优化器状态失败: {e}。优化器将使用初始状态。")
    elif optimizer and require_optimizer_scheduler:
        logging.warning("检查点中未找到优化器状态，但被要求加载。")

    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("学习率调度器状态加载完成。")
        except Exception as e:
            logging.warning(f"加载学习率调度器状态失败: {e}。调度器将使用初始状态。")
    elif scheduler and require_optimizer_scheduler and ('scheduler_state_dict' not in checkpoint or checkpoint['scheduler_state_dict'] is None):
        logging.warning("检查点中未找到调度器状态或状态为None，但被要求加载。")
    
    start_epoch = checkpoint.get('epoch', 0) 
    loaded_config_dict = checkpoint.get('config_dict', None)
    if loaded_config_dict is None:
        logging.warning("检查点中未找到 'config_dict'。")
    
    loaded_tags_list = checkpoint.get('tags_list', None)
    loaded_tag_to_idx = checkpoint.get('tag_to_idx', None)
    if loaded_tags_list is None or loaded_tag_to_idx is None:
        logging.warning("检查点中未找到词汇表 (tags_list/tag_to_idx)。")

    logging.info(f"检查点加载成功。将从 epoch {start_epoch} 开始。")
    return start_epoch, loaded_config_dict, loaded_tags_list, loaded_tag_to_idx


def calculate_metrics(preds_sigmoid, targets_binary, threshold=0.5, eps=1e-7):
    import warnings
    import numpy as np
    # import logging # logging 实例应从调用者传入或使用全局的
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, hamming_loss

    # 确保 preds_sigmoid 和 targets_binary 是 numpy 数组
    if hasattr(preds_sigmoid, 'detach'): # 如果是 PyTorch 张量
        preds_sigmoid_np = preds_sigmoid.detach().cpu().numpy()
    else: # 假设已经是 numpy 数组
        preds_sigmoid_np = np.array(preds_sigmoid)

    if hasattr(targets_binary, 'detach'):
        targets_binary_np = targets_binary.detach().cpu().numpy().astype(int)
    else:
        targets_binary_np = np.array(targets_binary).astype(int)


    # 分析标签统计信息 (这些日志可能比较冗余，可以考虑只在 rank 0 打印)
    # total_positive_targets = np.sum(targets_binary_np)
    # total_samples = targets_binary_np.shape[0]
    # total_classes = targets_binary_np.shape[1] if len(targets_binary_np.shape) > 1 else 1
    # positive_samples_per_class = np.sum(targets_binary_np, axis=0)
    # classes_with_positives = np.sum(positive_samples_per_class > 0)
    # logging.debug(f"验证集: {total_samples}张图片, {total_classes}个类别")
    # logging.debug(f"总正样本数: {total_positive_targets}, 平均每张图片{total_positive_targets/total_samples:.2f}个标签")
    # logging.debug(f"共有{classes_with_positives}/{total_classes}个类别具有至少一个正样本")
    
    preds_binary_np = (preds_sigmoid_np >= threshold).astype(int)
    # total_predicted_positives = np.sum(preds_binary_np)
    # predicted_classes_with_positives = np.sum(np.sum(preds_binary_np, axis=0) > 0)
    # logging.debug(f"阈值{threshold}下共预测{total_predicted_positives}个正样本, {predicted_classes_with_positives}个类别有预测")
    
    # if total_positive_targets == 0:
    #     logging.warning(f"验证集中没有正样本! 所有{total_classes}个类别都是负样本。")

    metrics = {}

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        targets_binary_np, preds_binary_np, average='micro', zero_division=0
    )
    metrics['precision_micro'] = precision_micro
    metrics['recall_micro'] = recall_micro
    metrics['f1_micro'] = f1_micro

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets_binary_np, preds_binary_np, average='macro', zero_division=0
    )
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    
    # ... (其他指标计算与原版类似) ...
    # 计算 mAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # 忽略所有 sklearn 警告，例如 "No positive samples in y_true"
        try:
            # 检查每个类别是否有至少一个正样本
            has_positive_samples_per_class = np.sum(targets_binary_np, axis=0) > 0
            
            if np.any(has_positive_samples_per_class): # 至少有一个类别有正样本
                metrics['mAP_macro'] = average_precision_score(targets_binary_np, preds_sigmoid_np, average='macro')
                metrics['mAP_micro'] = average_precision_score(targets_binary_np, preds_sigmoid_np, average='micro')
            else: # 没有一个类别有正样本
                # logging.warning("mAP 计算跳过：目标中没有正样本。")
                metrics['mAP_macro'] = 0.0
                metrics['mAP_micro'] = 0.0
        except ValueError as e: # 例如，当所有 y_true 都是负样本时
            # logging.warning(f"计算 mAP 时出错 (可能所有目标都是负样本): {e}。mAP 将设为0。")
            metrics['mAP_macro'] = 0.0
            metrics['mAP_micro'] = 0.0
        except Exception as e: # 其他意外错误
            logging.error(f"计算 mAP 时发生未知错误: {e}。mAP 将设为0。")
            metrics['mAP_macro'] = 0.0
            metrics['mAP_micro'] = 0.0


    metrics['hamming_loss'] = hamming_loss(targets_binary_np, preds_binary_np)
    exact_match_ratio = np.all(preds_binary_np == targets_binary_np, axis=1).mean()
    metrics['exact_match_ratio'] = exact_match_ratio

    return metrics
