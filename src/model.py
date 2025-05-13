import timm
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def get_model_classifier_info(model_instance):
    """
    辅助函数，获取模型的分类器层及其在 state_dict 中的名称前缀
    """
    classifier_layer = None
    classifier_key_prefix = None
    if hasattr(model_instance, 'get_classifier'):
        classifier_layer = model_instance.get_classifier()
        for name, mod in model_instance.named_modules():
            if mod is classifier_layer:
                classifier_key_prefix = name 
                if name: 
                    classifier_key_prefix += '.'
                break
    elif hasattr(model_instance, 'head') and isinstance(model_instance.head, nn.Linear):
        classifier_layer = model_instance.head
        classifier_key_prefix = 'head.'
    elif hasattr(model_instance, 'fc') and isinstance(model_instance.fc, nn.Linear):
        classifier_layer = model_instance.fc
        classifier_key_prefix = 'fc.'
    
    if classifier_layer is None and logger: # 检查 logger 是否已初始化
        logger.warning(f"无法自动识别模型 {type(model_instance).__name__} 的分类器层")
    
    return classifier_layer, classifier_key_prefix

def adapt_classifier_weights(target_model_with_new_classifier, 
                             pretrained_temp_model, 
                             new_tags_list, new_tag_to_idx,
                             old_tags_list_from_ckpt, old_tag_to_idx_from_ckpt):
    """
    适配分类器权重 (逻辑与原版相同)
    """
    new_classifier_layer, _ = get_model_classifier_info(target_model_with_new_classifier)
    old_classifier_layer, _ = get_model_classifier_info(pretrained_temp_model)

    if not new_classifier_layer or not old_classifier_layer:
        if logger: logger.warning("无法适配分类器权重：未能找到新模型或旧模型的分类器层，新分类器将使用默认初始化")
        return

    old_weights = old_classifier_layer.weight.data
    old_biases = old_classifier_layer.bias.data if old_classifier_layer.bias is not None else None
    new_weights = new_classifier_layer.weight.data
    new_biases = new_classifier_layer.bias.data if new_classifier_layer.bias is not None else None
    
    num_common_tags_copied = 0
    num_new_tags_in_classifier = 0

    with torch.no_grad():
        for new_idx, tag_name in enumerate(new_tags_list):
            if tag_name in old_tag_to_idx_from_ckpt:
                old_idx = old_tag_to_idx_from_ckpt[tag_name]
                if old_idx < old_weights.shape[0] and new_idx < new_weights.shape[0]:
                    new_weights[new_idx] = old_weights[old_idx]
                    if old_biases is not None and new_biases is not None and \
                       old_idx < old_biases.shape[0] and new_idx < new_biases.shape[0]:
                        new_biases[new_idx] = old_biases[old_idx]
                    num_common_tags_copied += 1
                else:
                    if logger: logger.warning(f"适配分类器权重时发生索引越界：标签 '{tag_name}', 旧索引 {old_idx}, 新索引 {new_idx}，该标签权重未复制")
            else:
                num_new_tags_in_classifier +=1

    if logger:
        logger.info(f"分类器权重适配完成：为 {num_common_tags_copied} 个共同标签复制了权重")
        logger.info(f"新分类器中包含 {num_new_tags_in_classifier} 个新增标签，其权重使用默认初始化")


def get_model(current_run_config, 
              old_checkpoint_config_dict=None, 
              old_checkpoint_state_dict=None,  
              current_tags_list=None, current_tag_to_idx=None, 
              old_tags_list_from_ckpt=None, old_tag_to_idx_from_ckpt=None
              ):
    """
    获取或创建模型实例。
    重要: 此函数现在返回在 CPU 上的模型。移动到特定 GPU 的操作由调用者 (train.py) 完成。
    """
    target_num_classes = current_run_config.NUM_CLASSES
    target_model_name = current_run_config.MODEL_NAME
    # rank = current_run_config.DEVICE # DEVICE 在 DDP 中是 rank
    # is_master_process = (rank == 0) # 用于控制日志输出

    if old_checkpoint_state_dict and old_checkpoint_config_dict:
        # 微调场景
        if logger: logger.info(f"进入微调模式，目标模型: {target_model_name}, 目标类别数: {target_num_classes}")
        
        old_model_name = old_checkpoint_config_dict.get('MODEL_NAME', target_model_name)
        old_num_classes = len(old_tags_list_from_ckpt) if old_tags_list_from_ckpt else old_checkpoint_config_dict.get('NUM_CLASSES', -1)

        if old_num_classes == -1:
            if logger: logger.error("微调错误：无法从旧检查点确定类别数 (NUM_CLASSES)")
            raise ValueError("旧检查点必须能提供类别数或词汇表信息")

        if logger: logger.info(f"旧检查点模型: {old_model_name}, 类别数: {old_num_classes}")

        target_model = timm.create_model(
            target_model_name,
            pretrained=False, 
            num_classes=target_num_classes,
            drop_rate=current_run_config.DROP_RATE,
            drop_path_rate=current_run_config.DROP_PATH_RATE,
        ) # 模型在 CPU 上创建
        target_model_dict = target_model.state_dict()

        temp_model_for_loading_old_weights = timm.create_model(
            old_model_name, 
            pretrained=False,
            num_classes=old_num_classes, 
        ) # 模型在 CPU 上创建
        
        if any(key.startswith('module.') for key in old_checkpoint_state_dict.keys()):
            old_checkpoint_state_dict = {k.replace('module.', ''): v for k, v in old_checkpoint_state_dict.items()}
        
        load_status = temp_model_for_loading_old_weights.load_state_dict(old_checkpoint_state_dict, strict=False)
        if (load_status.missing_keys or load_status.unexpected_keys) and logger:
            logger.warning(f"加载旧检查点到临时模型时遇到状态不匹配: Missing={load_status.missing_keys}, Unexpected={load_status.unexpected_keys}")
        if logger: logger.info("已将旧检查点权重加载到临时模型 (CPU)")

        _, old_classifier_prefix = get_model_classifier_info(temp_model_for_loading_old_weights)
        
        num_backbone_weights_copied = 0
        for key, value in temp_model_for_loading_old_weights.state_dict().items():
            if old_classifier_prefix and key.startswith(old_classifier_prefix):
                continue 
            if key in target_model_dict and target_model_dict[key].shape == value.shape:
                target_model_dict[key] = value
                num_backbone_weights_copied += 1
        
        target_model.load_state_dict(target_model_dict, strict=False)
        if logger: logger.info(f"已从临时模型复制 {num_backbone_weights_copied} 个 backbone 相关权重到目标模型 (CPU)")

        if old_tags_list_from_ckpt and current_tags_list and (old_num_classes != target_num_classes or old_tags_list_from_ckpt != current_tags_list):
            if logger: logger.info("检测到词汇表或类别数发生变化，开始适配分类器权重...")
            adapt_classifier_weights(
                target_model_with_new_classifier=target_model,
                pretrained_temp_model=temp_model_for_loading_old_weights,
                new_tags_list=current_tags_list, new_tag_to_idx=current_tag_to_idx,
                old_tags_list_from_ckpt=old_tags_list_from_ckpt, old_tag_to_idx_from_ckpt=old_tag_to_idx_from_ckpt
            )
        elif old_num_classes == target_num_classes:
             if logger: logger.info("类别数未变，将尝试直接加载旧分类器权重（如果适用）")
             if old_classifier_prefix:
                old_classifier_weights = {
                    key: val for key, val in temp_model_for_loading_old_weights.state_dict().items() 
                    if key.startswith(old_classifier_prefix)
                }
                current_classifier_weights_loaded = 0
                temp_target_sd = target_model.state_dict()
                for k, v_old in old_classifier_weights.items():
                    if k in temp_target_sd and temp_target_sd[k].shape == v_old.shape:
                        temp_target_sd[k] = v_old
                        current_classifier_weights_loaded +=1
                target_model.load_state_dict(temp_target_sd, strict=False)
                if logger: logger.info(f"类别数未变，已加载 {current_classifier_weights_loaded} 个旧分类器权重 (CPU)")

        del temp_model_for_loading_old_weights 
        return target_model # 返回 CPU 上的模型

    else: # 从头训练 或 继续训练
        if logger: logger.info(f"创建新模型 (或用于继续训练): {target_model_name}, 类别数: {target_num_classes}, ImageNet预训练: {current_run_config.PRETRAINED}")
        
        use_imagenet_pretrained = current_run_config.PRETRAINED
        if old_checkpoint_state_dict: 
            use_imagenet_pretrained = False

        model = timm.create_model(
            target_model_name,
            pretrained=use_imagenet_pretrained,
            num_classes=target_num_classes, 
            drop_rate=current_run_config.DROP_RATE,
            drop_path_rate=current_run_config.DROP_PATH_RATE,
        ) # 模型在 CPU 上创建
        
        return model # 返回 CPU 上的模型
