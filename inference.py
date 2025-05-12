import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import logging
import glob
from contextlib import nullcontext

from src.model import get_model
from src.utils import load_checkpoint, setup_logger
from configs.base_config import BaseConfig

logger = None

def predict_single_image(model_instance, image_path, transform, device, tags_list_from_ckpt, inference_threshold):
    """对单张图片进行预测"""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"错误: 图像文件 {image_path} 未找到")
        return None
    except Exception as e:
        logger.error(f"加载图像 {image_path} 时出错: {e}")
        return None
    
    img_tensor = transform(image).unsqueeze(0).to(device) # 增加 batch 维度并移动到设备

    # 尝试启用 SDPA
    sdp_context = nullcontext()
    if hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0": # 检查 torch.nn.attention
        try:
            # 更新为新的 API
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
        except Exception:
            pass # 推理时静默失败

    with sdp_context:
        with torch.no_grad(): # 推理时不需要计算梯度
            logits = model_instance(img_tensor)
            probabilities = torch.sigmoid(logits).squeeze() # 移除 batch 维度

    # 获取高于阈值的标签及其置信度
    predicted_tags_with_confidence = []
    for i, prob in enumerate(probabilities):
        if prob.item() > inference_threshold:
            if i < len(tags_list_from_ckpt): # 确保索引有效
                predicted_tags_with_confidence.append((tags_list_from_ckpt[i], prob.item()))
            else:
                logger.warning(f"警告: 预测索引 {i} 超出词汇表范围 (大小: {len(tags_list_from_ckpt)})")

    # 按置信度降序排列
    predicted_tags_with_confidence.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_tags_with_confidence


def main(args):
    global logger
    logger = setup_logger(os.path.join(os.path.dirname(args.checkpoint_path), "inference_logs"), "inference_run")

    if not os.path.isfile(args.checkpoint_path):
        logger.error(f"错误: 检查点文件未找到于 {args.checkpoint_path}")
        return

    # 1. 加载检查点信息
    # 先加载到 CPU，避免直接在 GPU 上分配过多内存
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    loaded_config_dict = checkpoint.get('config_dict')
    tags_list_from_ckpt = checkpoint.get('tags_list')
    # tag_to_idx_from_ckpt = checkpoint.get('tag_to_idx') # 推理时主要用 tags_list

    if not loaded_config_dict or not tags_list_from_ckpt:
        logger.error("错误: 检查点中缺少配置信息 (config_dict) 或词汇表 (tags_list)")
        return

    # 2. 构建/恢复配置对象
    # 使用检查点中的配置为主，但允许命令行覆盖设备和推理阈值
    inference_config = BaseConfig.from_dict(loaded_config_dict)
    inference_config.DEVICE = args.device if args.device else inference_config.DEVICE
    # 如果命令行指定了阈值，则覆盖配置中的阈值
    if args.threshold is not None:
        inference_config.INFERENCE_THRESHOLD = args.threshold
    
    logger.info(f"使用设备: {inference_config.DEVICE}")
    logger.info(f"推理阈值: {inference_config.INFERENCE_THRESHOLD}")
    logger.info(f"模型名称 (来自检查点配置): {inference_config.MODEL_NAME}")
    logger.info(f"图像尺寸 (来自检查点配置): {inference_config.IMAGE_SIZE}")
    logger.info(f"词汇表大小 (来自检查点): {len(tags_list_from_ckpt)}")
    
    # 确保 NUM_CLASSES 与加载的词汇表一致
    inference_config.NUM_CLASSES = len(tags_list_from_ckpt)
    inference_config.PRETRAINED = False # 加载自己的权重，不需要 ImageNet 预训练

    # 3. 初始化模型
    model = get_model(inference_config) # get_model 根据配置创建模型结构
    
    # 加载模型权重 (utils.load_checkpoint 负责处理 'module.' 前缀等)
    # 对于推理，我们只需要加载模型权重，不需要优化器和调度器
    _, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer=None, scheduler=None, device=inference_config.DEVICE)
    model.eval() # 设置为评估模式

    # 4. 定义图像变换
    # 推理时使用与验证集相同的变换 (通常不包含数据增强)
    transform = transforms.Compose([
        transforms.Resize((inference_config.IMAGE_SIZE, inference_config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=inference_config.NORM_MEAN, std=inference_config.NORM_STD),
    ])

    # 5. 处理输入 (单张图片或图片文件夹)
    image_paths_to_predict = []
    if os.path.isfile(args.input_path):
        image_paths_to_predict.append(args.input_path)
    elif os.path.isdir(args.input_path):
        for ext in inference_config.IMAGE_EXTENSIONS: # 使用配置中定义的扩展名
            image_paths_to_predict.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        if not image_paths_to_predict:
            logger.warning(f"在目录 {args.input_path} 中未找到支持的图像文件")
            return
    else:
        logger.error(f"错误: 输入路径 {args.input_path} 不是有效的文件或目录")
        return

    logger.info(f"共找到 {len(image_paths_to_predict)} 张图片进行推理")

    # 6. 逐张图片进行预测并输出
    for img_path in image_paths_to_predict:
        logger.info(f"正在处理图片: {img_path}")
        predicted_results = predict_single_image(
            model, img_path, transform, inference_config.DEVICE, 
            tags_list_from_ckpt, inference_config.INFERENCE_THRESHOLD
        )
        
        print(f"\n图片: {os.path.basename(img_path)}") # 只打印文件名
        if predicted_results is not None:
            if predicted_results:
                print(f"预测标签 (置信度 > {inference_config.INFERENCE_THRESHOLD}):")
                for tag, confidence in predicted_results:
                    print(f"{tag}: {confidence:.4f}") # 按要求的格式输出
            else:
                print(f"没有标签的置信度高于阈值 {inference_config.INFERENCE_THRESHOLD}")
        else:
            print("图片处理失败")
        print("-" * 30) # 分隔符

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 图像多标签分类推理脚本")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="训练好的模型检查点文件路径 (.pth.tar)")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="需要进行推理的单张图片路径或包含多张图片的文件夹路径")
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'],
                        help="覆盖检查点配置中的 DEVICE 设置 (例如，在无GPU环境强制使用cpu)")
    parser.add_argument("--threshold", type=float, default=None, 
                        help="覆盖检查点配置中的 INFERENCE_THRESHOLD，用于过滤输出标签的置信度")
    
    args = parser.parse_args()
    
    # 设置基础日志，以便在 logger 初始化前捕获错误
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    main(args)
