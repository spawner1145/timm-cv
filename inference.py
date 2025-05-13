import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
import logging # 保持 logging 导入
import glob
from contextlib import nullcontext

from src.model import get_model
from src.utils import load_checkpoint, setup_logger # setup_logger 用于创建特定 logger
from configs.base_config import BaseConfig

# 全局 logger 变量，将在 main 中被赋值为 setup_logger 返回的实例
logger = None

def predict_single_image(model_instance, image_path, transform, device, tags_list_from_ckpt, inference_threshold):
    """对单张图片进行预测"""
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        logger.error(f"错误: 图像文件 {image_path} 未找到") # 使用我们配置的 logger
        return None
    except Exception as e:
        logger.error(f"加载图像 {image_path} 时出错: {e}") # 使用我们配置的 logger
        return None
    
    img_tensor = transform(image).unsqueeze(0).to(device) 

    sdp_context = nullcontext()
    if hasattr(torch.nn.attention, 'sdpa_kernel') and torch.__version__ >= "2.0.0": 
        try:
            sdp_context = torch.nn.attention.sdpa_kernel([
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                torch.nn.attention.SDPBackend.MATH
            ])
        except Exception:
            pass 

    with sdp_context:
        with torch.no_grad(): 
            logits = model_instance(img_tensor)
            probabilities = torch.sigmoid(logits).squeeze() 

    predicted_tags_with_confidence = []
    for i, prob in enumerate(probabilities):
        if prob.item() > inference_threshold:
            if i < len(tags_list_from_ckpt): 
                predicted_tags_with_confidence.append((tags_list_from_ckpt[i], prob.item()))
            else:
                # 使用我们配置的 logger
                logger.warning(f"警告: 预测索引 {i} 超出词汇表范围 (大小: {len(tags_list_from_ckpt)})")

    predicted_tags_with_confidence.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_tags_with_confidence


def main(args):
    global logger # 声明使用全局 logger 变量
    
    # 配置一个特定的 logger 实例供此脚本使用
    # 日志文件名将基于检查点路径的目录
    log_dir = os.path.join(os.path.dirname(args.checkpoint_path) if args.checkpoint_path else ".", "inference_logs")
    run_name_prefix = os.path.splitext(os.path.basename(args.checkpoint_path))[0] if args.checkpoint_path else "inference"
    logger = setup_logger(log_dir, f"{run_name_prefix}_inference")


    if not os.path.isfile(args.checkpoint_path):
        logger.error(f"错误: 检查点文件未找到于 {args.checkpoint_path}")
        return

    # 1. 加载检查点信息 (先加载到 CPU)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    
    loaded_config_dict = checkpoint.get('config_dict')
    tags_list_from_ckpt = checkpoint.get('tags_list')

    if not loaded_config_dict or not tags_list_from_ckpt:
        logger.error("错误: 检查点中缺少配置信息 (config_dict) 或词汇表 (tags_list)")
        return

    # 2. 构建/恢复配置对象
    inference_config = BaseConfig.from_dict(loaded_config_dict)
    
    # 确定推理设备
    if args.device: # 如果命令行指定了设备
        target_device_str = args.device
    elif torch.cuda.is_available() and isinstance(inference_config.DEVICE, int): # 如果配置中是GPU rank且CUDA可用
        target_device_str = f"cuda:{inference_config.DEVICE}"
    elif torch.cuda.is_available() and inference_config.DEVICE == "cuda": # 如果配置是 "cuda" 且CUDA可用
         target_device_str = "cuda"
    else: # 其他情况（如配置是 "cpu"，或CUDA不可用）
        target_device_str = "cpu"
    
    # 将字符串设备转换为 torch.device 对象
    final_device = torch.device(target_device_str)
    inference_config.DEVICE = final_device # 更新配置中的DEVICE为torch.device对象

    if args.threshold is not None:
        inference_config.INFERENCE_THRESHOLD = args.threshold
    
    logger.info(f"使用设备: {inference_config.DEVICE}")
    logger.info(f"推理阈值: {inference_config.INFERENCE_THRESHOLD}")
    logger.info(f"模型名称 (来自检查点配置): {inference_config.MODEL_NAME}")
    logger.info(f"图像尺寸 (来自检查点配置): {inference_config.IMAGE_SIZE}")
    logger.info(f"词汇表大小 (来自检查点): {len(tags_list_from_ckpt)}")
    
    inference_config.NUM_CLASSES = len(tags_list_from_ckpt)
    inference_config.PRETRAINED = False 

    # 3. 初始化模型 (在 CPU 上)
    model = get_model(inference_config) 
    
    # 4. 加载模型权重 (load_checkpoint 会将权重加载到模型中，但模型本身仍在CPU)
    #    load_checkpoint 内部的 map_location=inference_config.DEVICE 确保权重张量在加载时被映射到目标设备
    #    但 model 对象本身需要显式移动。
    #    为了清晰，我们先加载权重，再移动整个模型。
    _, _, _, _ = load_checkpoint(args.checkpoint_path, model, optimizer=None, scheduler=None, device=inference_config.DEVICE)
    
    # 5. ***关键步骤：将整个模型移动到目标设备***
    model.to(inference_config.DEVICE)
    model.eval() 

    # 6. 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((inference_config.IMAGE_SIZE, inference_config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=inference_config.NORM_MEAN, std=inference_config.NORM_STD),
    ])

    # 7. 处理输入
    image_paths_to_predict = []
    if os.path.isfile(args.input_path):
        image_paths_to_predict.append(args.input_path)
    elif os.path.isdir(args.input_path):
        for ext in inference_config.IMAGE_EXTENSIONS: 
            image_paths_to_predict.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
        if not image_paths_to_predict:
            logger.warning(f"在目录 {args.input_path} 中未找到支持的图像文件")
            return
    else:
        logger.error(f"错误: 输入路径 {args.input_path} 不是有效的文件或目录")
        return

    logger.info(f"共找到 {len(image_paths_to_predict)} 张图片进行推理")

    # 8. 逐张图片进行预测并输出
    for img_path in image_paths_to_predict:
        logger.info(f"正在处理图片: {img_path}")
        predicted_results = predict_single_image(
            model, img_path, transform, inference_config.DEVICE, 
            tags_list_from_ckpt, inference_config.INFERENCE_THRESHOLD
        )
        
        # 使用 print 输出结果到控制台，logger 用于记录过程信息
        print(f"\n图片: {os.path.basename(img_path)}") 
        if predicted_results is not None:
            if predicted_results:
                print(f"预测标签 (置信度 > {inference_config.INFERENCE_THRESHOLD}):")
                for tag, confidence in predicted_results:
                    print(f"  {tag}: {confidence:.4f}")
            else:
                print(f"  没有标签的置信度高于阈值 {inference_config.INFERENCE_THRESHOLD}")
        else:
            print("  图片处理失败")
        print("-" * 30) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru 图像多标签分类推理脚本")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="训练好的模型检查点文件路径 (.pth.tar)")
    parser.add_argument("--input_path", type=str, required=True, 
                        help="需要进行推理的单张图片路径或包含多张图片的文件夹路径")
    parser.add_argument("--device", type=str, default=None, # 允许用户覆盖设备
                        help="指定推理设备 (例如 'cuda:0', 'cpu')。如果未指定，则尝试使用检查点配置中的设备或CUDA（如果可用）。")
    parser.add_argument("--threshold", type=float, default=None, 
                        help="覆盖检查点配置中的 INFERENCE_THRESHOLD，用于过滤输出标签的置信度")
    
    args = parser.parse_args()
    
    # 不再使用 logging.basicConfig()，完全依赖 setup_logger 来配置日志
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    main(args)
