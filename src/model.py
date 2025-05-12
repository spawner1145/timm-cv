import timm
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

def get_model_classifier_info(model_instance):
    """
    辅助函数，获取模型的分类器层及其在 state_dict 中的名称前缀
    Args:
        model_instance: timm 创建的模型实例
    Returns:
        (classifier_layer, classifier_key_prefix)
        如果找不到，则 classifier_layer 为 None
    """
    classifier_layer = None
    classifier_key_prefix = None

    # 尝试 timm 的通用方法
    if hasattr(model_instance, 'get_classifier'):
        classifier_layer = model_instance.get_classifier()
        # 查找此层在模型中的名称，用于构建 state_dict 的键
        for name, mod in model_instance.named_modules():
            if mod is classifier_layer:
                classifier_key_prefix = name # 例如 "head" 或 "fc"
                if name: # 如果不是顶层模块，则加上点号
                    classifier_key_prefix += '.'
                break
    # 针对特定属性名 (timm 中常见)
    elif hasattr(model_instance, 'head') and isinstance(model_instance.head, nn.Linear):
        classifier_layer = model_instance.head
        classifier_key_prefix = 'head.'
    elif hasattr(model_instance, 'fc') and isinstance(model_instance.fc, nn.Linear):
        classifier_layer = model_instance.fc
        classifier_key_prefix = 'fc.'
    
    if classifier_layer is None:
        logger.warning(f"无法自动识别模型 {type(model_instance).__name__} 的分类器层")
    
    return classifier_layer, classifier_key_prefix

def adapt_classifier_weights(target_model_with_new_classifier, # 目标模型，已更新分类头
                             pretrained_temp_model, # 加载了旧checkpoint的临时模型
                             new_tags_list, new_tag_to_idx,
                             old_tags_list_from_ckpt, old_tag_to_idx_from_ckpt):
    """
    适配分类器权重，当微调时标签集发生变化（增/删标签），此函数尝试：
    1. 对于新旧标签集中的共同标签，从旧模型复制权重到新模型
    2. 新模型中新增的标签，其分类器权重保持默认初始化（通常是 Kaiming 初始化）
    """
    new_classifier_layer, _ = get_model_classifier_info(target_model_with_new_classifier)
    old_classifier_layer, _ = get_model_classifier_info(pretrained_temp_model)

    if not new_classifier_layer or not old_classifier_layer:
        logger.warning("无法适配分类器权重：未能找到新模型或旧模型的分类器层，新分类器将使用默认初始化")
        return

    # 获取旧分类器的权重和偏置 (如果存在)
    old_weights = old_classifier_layer.weight.data
    old_biases = old_classifier_layer.bias.data if old_classifier_layer.bias is not None else None

    # 获取新分类器的权重和偏置 (如果存在)
    # 注意：此时新分类器的权重已经是 timm.create_model 或 reset_classifier 后的默认初始化状态
    new_weights = new_classifier_layer.weight.data
    new_biases = new_classifier_layer.bias.data if new_classifier_layer.bias is not None else None
    
    num_common_tags_copied = 0
    num_new_tags_in_classifier = 0 # 仅存在于新词汇表的标签

    with torch.no_grad(): # 确保操作不被跟踪梯度
        for new_idx, tag_name in enumerate(new_tags_list):
            if tag_name in old_tag_to_idx_from_ckpt: # 如果是共同标签
                old_idx = old_tag_to_idx_from_ckpt[tag_name]
                
                # 检查索引是否越界 (理论上不应发生，因为是基于词汇表构建的)
                if old_idx < old_weights.shape[0] and new_idx < new_weights.shape[0]:
                    new_weights[new_idx] = old_weights[old_idx]
                    if old_biases is not None and new_biases is not None and \
                       old_idx < old_biases.shape[0] and new_idx < new_biases.shape[0]:
                        new_biases[new_idx] = old_biases[old_idx]
                    num_common_tags_copied += 1
                else:
                    logger.warning(f"适配分类器权重时发生索引越界：标签 '{tag_name}', 旧索引 {old_idx}, 新索引 {new_idx}，该标签权重未复制")
            else: # 如果是新词汇表中新增的标签
                num_new_tags_in_classifier +=1
                # 新增标签的权重保持其在 new_classifier_layer 创建时的默认初始化值

    logger.info(f"分类器权重适配完成：为 {num_common_tags_copied} 个共同标签复制了权重")
    logger.info(f"新分类器中包含 {num_new_tags_in_classifier} 个新增标签，其权重使用默认初始化")


def get_model(current_run_config, # 当前运行的配置对象
              # 以下参数仅在微调 (finetune) 模式下，且词汇表可能变化时使用
              old_checkpoint_config_dict=None, # 从旧检查点加载的配置字典
              old_checkpoint_state_dict=None,  # 旧检查点的模型 state_dict
              current_tags_list=None, current_tag_to_idx=None, # 当前运行的词汇表
              old_tags_list_from_ckpt=None, old_tag_to_idx_from_ckpt=None # 旧检查点的词汇表
              ):
    """
    获取或创建模型实例
    - 从头训练: 根据 current_run_config 创建新模型
    - 继续训练 (resume): current_run_config 的 NUM_CLASSES 应与检查点匹配，直接加载权重
    - 微调 (finetune):
        - 如果词汇表不变: 加载旧模型权重，分类头不变
        - 如果词汇表变化:
            1. 根据 current_run_config (新的 NUM_CLASSES) 创建模型结构
            2. 加载旧检查点的 backbone 权重
            3. 调用 adapt_classifier_weights 迁移共同标签的分类器权重
    """
    target_num_classes = current_run_config.NUM_CLASSES
    target_model_name = current_run_config.MODEL_NAME

    if old_checkpoint_state_dict and old_checkpoint_config_dict:
        # 微调场景
        logger.info(f"进入微调模式，目标模型: {target_model_name}, 目标类别数: {target_num_classes}")
        
        old_model_name = old_checkpoint_config_dict.get('MODEL_NAME', target_model_name) # 如果旧配置没存MODEL_NAME，则假设一致
        old_num_classes = len(old_tags_list_from_ckpt) if old_tags_list_from_ckpt else old_checkpoint_config_dict.get('NUM_CLASSES', -1)

        if old_num_classes == -1:
            logger.error("微调错误：无法从旧检查点确定类别数 (NUM_CLASSES)")
            raise ValueError("旧检查点必须能提供类别数或词汇表信息")

        logger.info(f"旧检查点模型: {old_model_name}, 类别数: {old_num_classes}")

        # 1. 创建目标模型结构 (使用新的类别数)
        #    timm.create_model 会自动初始化新分类头的权重 (通常是 Kaiming 初始化)
        target_model = timm.create_model(
            target_model_name,
            pretrained=False, # 因为我们将从自己的检查点加载权重
            num_classes=target_num_classes, # 使用新的类别数
            drop_rate=current_run_config.DROP_RATE,
            drop_path_rate=current_run_config.DROP_PATH_RATE,
        )
        target_model_dict = target_model.state_dict()

        # 2. 创建一个临时模型 (使用旧的类别数) 以便完整加载旧检查点的权重
        #    这确保了即使分类头尺寸不同，也能正确加载 backbone
        temp_model_for_loading_old_weights = timm.create_model(
            old_model_name, # 使用旧模型的名称
            pretrained=False,
            num_classes=old_num_classes, # 使用旧模型的类别数
        )
        
        # 处理 'module.' 前缀 (如果旧模型是用 DataParallel/DDP 保存的)
        if any(key.startswith('module.') for key in old_checkpoint_state_dict.keys()):
            old_checkpoint_state_dict = {k.replace('module.', ''): v for k, v in old_checkpoint_state_dict.items()}
        
        # 加载旧权重到临时模型
        # strict=False 是因为如果模型结构略有不同（例如，timm版本更新导致层名变化），这能更宽容地加载
        # 但主要目的是加载与分类头无关的部分
        load_status = temp_model_for_loading_old_weights.load_state_dict(old_checkpoint_state_dict, strict=False)
        if load_status.missing_keys or load_status.unexpected_keys:
            logger.warning(f"加载旧检查点到临时模型时遇到状态不匹配: Missing={load_status.missing_keys}, Unexpected={load_status.unexpected_keys}")
        logger.info("已将旧检查点权重加载到临时模型")

        # 3. 将 backbone 权重从临时模型复制到目标模型
        _, old_classifier_prefix = get_model_classifier_info(temp_model_for_loading_old_weights)
        
        num_backbone_weights_copied = 0
        for key, value in temp_model_for_loading_old_weights.state_dict().items():
            if old_classifier_prefix and key.startswith(old_classifier_prefix):
                continue # 跳过旧分类器的权重
            if key in target_model_dict and target_model_dict[key].shape == value.shape:
                target_model_dict[key] = value
                num_backbone_weights_copied += 1
            # else:
                # logger.debug(f"微调时跳过或形状不匹配的权重: {key}")
        
        target_model.load_state_dict(target_model_dict, strict=False) # strict=False 因为分类器可能还未完全匹配
        logger.info(f"已从临时模型复制 {num_backbone_weights_copied} 个 backbone 相关权重到目标模型")

        # 4. 如果词汇表发生变化，适配分类器权重
        if old_tags_list_from_ckpt and current_tags_list and (old_num_classes != target_num_classes or old_tags_list_from_ckpt != current_tags_list):
            logger.info("检测到词汇表或类别数发生变化，开始适配分类器权重...")
            adapt_classifier_weights(
                target_model_with_new_classifier=target_model,
                pretrained_temp_model=temp_model_for_loading_old_weights,
                new_tags_list=current_tags_list, new_tag_to_idx=current_tag_to_idx,
                old_tags_list_from_ckpt=old_tags_list_from_ckpt, old_tag_to_idx_from_ckpt=old_tag_to_idx_from_ckpt
            )
        elif old_num_classes == target_num_classes:
             # 类别数未变，但仍需加载分类器权重 (如果backbone复制时跳过了)
             # 这种情况理论上应该在上面的 backbone 权重复制中被覆盖，除非分类器名不同
             logger.info("类别数未变，将尝试直接加载旧分类器权重（如果适用）")
             if old_classifier_prefix: # 确保旧分类器前缀有效
                old_classifier_weights = {
                    key: val for key, val in temp_model_for_loading_old_weights.state_dict().items() 
                    if key.startswith(old_classifier_prefix)
                }
                # 更新目标模型的state_dict
                current_classifier_weights_loaded = 0
                temp_target_sd = target_model.state_dict()
                for k, v_old in old_classifier_weights.items():
                    if k in temp_target_sd and temp_target_sd[k].shape == v_old.shape:
                        temp_target_sd[k] = v_old
                        current_classifier_weights_loaded +=1
                target_model.load_state_dict(temp_target_sd, strict=False)
                logger.info(f"类别数未变，已加载 {current_classifier_weights_loaded} 个旧分类器权重")


        del temp_model_for_loading_old_weights # 释放临时模型占用的内存
        return target_model.to(current_run_config.DEVICE)

    else:
        # 从头训练 或 继续训练 (resume, 此时 NUM_CLASSES 应已从检查点正确设置)
        logger.info(f"创建新模型 (或用于继续训练): {target_model_name}, 类别数: {target_num_classes}, ImageNet预训练: {current_run_config.PRETRAINED}")
        
        # 如果是继续训练 (args.resume_checkpoint 非空), PRETRAINED 应为 False，因为权重将从检查点加载
        # 这个逻辑在 train.py 中处理：如果 resume, PRETRAINED 会被忽略，权重从 checkpoint 加载
        use_imagenet_pretrained = current_run_config.PRETRAINED
        if old_checkpoint_state_dict: # 这意味着是 resume 场景，不应再用 imagenet 预训练
            use_imagenet_pretrained = False

        model = timm.create_model(
            target_model_name,
            pretrained=use_imagenet_pretrained,
            num_classes=target_num_classes, # 确保使用正确的类别数
            drop_rate=current_run_config.DROP_RATE,
            drop_path_rate=current_run_config.DROP_PATH_RATE,
        )
        
        # 如果是继续训练，权重将在 train.py 中通过 load_checkpoint 加载
        return model.to(current_run_config.DEVICE)

