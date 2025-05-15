import os
import torch
import logging

def setup_logger(log_dir, run_name, log_level=logging.INFO):
    """配置日志记录器，同时输出到控制台和文件。"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(run_name)
    logger.setLevel(log_level)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    #file_handler = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), mode='a', encoding='utf-8')
    #file_handler.setFormatter(formatter)
    #logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

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
                    tags_list, tag_to_idx, 
                    filename_prefix="checkpoint", # train.py会传入如 "run_name_epoch_X" 或 "run_name_best"
                    is_best_marker_for_log=False, # 重命名参数以明确其主要用途
                    output_dir="trained_models/"):
    """
    保存模型检查点。
    文件名现在完全由 filename_prefix 控制。
    is_best_marker_for_log 参数仅用于在日志中添加标记。
    """
    state = {
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'tags_list': tags_list,
        'tag_to_idx': tag_to_idx,
        'is_best_model_marker': is_best_marker_for_log # 可选：在检查点内部存储是否为最佳的标记
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

    target_filename = f"{filename_prefix}.pth.tar" # 文件名由传入的 prefix 决定
    filepath = os.path.join(output_dir, target_filename)
    
    try:
        torch.save(state, filepath)
        log_message = f"检查点已保存到: {filepath} (Epoch {epoch})"
        if is_best_marker_for_log: # 如果调用者标记这是最佳模型
            log_message += " [标记为当前最佳]"
        logging.info(log_message)
    except Exception as e:
        logging.error(f"保存检查点 {filepath} 时出错: {e}")


def load_checkpoint(checkpoint_path, model_instance, 
                    optimizer=None, scheduler=None, device="cuda",
                    require_optimizer_scheduler=False): 
    if not os.path.isfile(checkpoint_path):
        logging.error(f"错误: 检查点文件未找到于 {checkpoint_path}")
        raise FileNotFoundError(f"检查点文件未找到于 {checkpoint_path}")
    
    logging.info(f"正在从检查点加载: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    
    model_state_dict = checkpoint['model_state_dict']
    if any(key.startswith('module.') for key in model_state_dict.keys()):
        logging.info("检测到 'module.' 前缀，正在移除...")
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
    
    load_status = model_instance.load_state_dict(model_state_dict, strict=False)
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
    import logging
    from sklearn.metrics import precision_recall_fscore_support, average_precision_score, hamming_loss

    preds_sigmoid_np = preds_sigmoid.detach().cpu().numpy()
    targets_binary_np = targets_binary.detach().cpu().numpy().astype(int) 

    # 分析标签统计信息
    total_positive_targets = np.sum(targets_binary_np)
    total_samples = targets_binary_np.shape[0]
    total_classes = targets_binary_np.shape[1] if len(targets_binary_np.shape) > 1 else 1
    
    # 计算每个类别的正样本数
    positive_samples_per_class = np.sum(targets_binary_np, axis=0)
    classes_with_positives = np.sum(positive_samples_per_class > 0)
    
    logging.debug(f"验证集: {total_samples}张图片, {total_classes}个类别")
    logging.debug(f"总正样本数: {total_positive_targets}, 平均每张图片{total_positive_targets/total_samples:.2f}个标签")
    logging.debug(f"共有{classes_with_positives}/{total_classes}个类别具有至少一个正样本")
    
    # 分析预测值分布
    pred_mean = np.mean(preds_sigmoid_np)
    pred_median = np.median(preds_sigmoid_np)
    pred_max = np.max(preds_sigmoid_np)
    
    logging.debug(f"预测概率: 均值={pred_mean:.4f}, 中位数={pred_median:.4f}, 最大值={pred_max:.4f}")
    
    # 计算预测的二值化结果
    preds_binary_np = (preds_sigmoid_np >= threshold).astype(int)
    total_predicted_positives = np.sum(preds_binary_np)
    predicted_classes_with_positives = np.sum(np.sum(preds_binary_np, axis=0) > 0)
    
    logging.debug(f"阈值{threshold}下共预测{total_predicted_positives}个正样本, {predicted_classes_with_positives}个类别有预测")
    
    if total_positive_targets == 0:
        logging.warning(f"验证集中没有正样本! 所有{total_classes}个类别都是负样本。这可能是由于标签过滤过于严格。")

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
    
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        targets_binary_np, preds_binary_np, average='samples', zero_division=0
    )
    metrics['precision_samples'] = precision_samples
    metrics['recall_samples'] = recall_samples
    metrics['f1_samples'] = f1_samples

    # 计算 mAP，忽略警告并处理无正样本的情况
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")
        try:
            # 检查每个类别是否有至少一个正样本
            has_positive_samples = np.sum(targets_binary_np, axis=0) > 0
            if np.all(has_positive_samples):
                # 所有类别都有正样本，正常计算
                metrics['mAP_macro'] = average_precision_score(targets_binary_np, preds_sigmoid_np, average='macro')
                metrics['mAP_micro'] = average_precision_score(targets_binary_np, preds_sigmoid_np, average='micro')
            else:
                # 只对有正样本的类别计算 mAP
                valid_classes = np.where(has_positive_samples)[0]
                if len(valid_classes) > 0:
                    logging.debug(f"只有{len(valid_classes)}/{total_classes}个类别有正样本，只对这些类别计算mAP")
                    metrics['mAP_macro'] = average_precision_score(
                        targets_binary_np[:, valid_classes], 
                        preds_sigmoid_np[:, valid_classes], 
                        average='macro'
                    )
                    # 微平均仍然可以在所有类别上计算
                    metrics['mAP_micro'] = average_precision_score(targets_binary_np, preds_sigmoid_np, average='micro')
                else:
                    # 极端情况：当前批次没有任何正样本
                    metrics['mAP_macro'] = 0.0
                    metrics['mAP_micro'] = 0.0
        except Exception as e:
            logging.debug(f"计算 mAP 时出错: {e}。mAP 将设为0。")
            metrics['mAP_macro'] = 0.0
            metrics['mAP_micro'] = 0.0

    metrics['hamming_loss'] = hamming_loss(targets_binary_np, preds_binary_np)
    exact_match_ratio = np.all(preds_binary_np == targets_binary_np, axis=1).mean()
    metrics['exact_match_ratio'] = exact_match_ratio

    return metrics
