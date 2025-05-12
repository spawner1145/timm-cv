"""
测试用脚本，分析数据集中标签的分布情况，以便更好地理解训练和验证中的正样本分布
"""
import os
import glob
import pandas as pd
import random
import logging
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_dataset(
    images_dir="data/images", 
    selected_tags_csv="data/selected_tags.csv",
    tag_separator=',',
    threshold=5,
    train_ratio=0.8
):
    """分析数据集中标签的分布情况"""
    
    # 1. 读取所有图像-文本文件对
    file_pairs = []
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        for img_path in glob.glob(os.path.join(images_dir, f"*{ext}")):
            base_filename, _ = os.path.splitext(img_path)
            txt_path = base_filename + ".txt"
            if os.path.exists(txt_path):
                file_pairs.append((img_path, txt_path))
    
    logger.info(f"找到 {len(file_pairs)} 个图像-文本文件对")

    # 2. 读取selected_tags.csv中的标签和出现次数
    selected_tags_df = pd.read_csv(selected_tags_csv)
    if 'count' not in selected_tags_df.columns:
        logger.warning("selected_tags.csv中没有'count'列，使用name列的长度作为默认")
        
    # 应用阈值过滤
    if 'count' in selected_tags_df.columns:
        selected_tags_df['count'] = pd.to_numeric(selected_tags_df['count'], errors='coerce')
        filtered_df = selected_tags_df[selected_tags_df['count'] >= threshold]
        valid_tags = set(filtered_df['name'].astype(str).unique())
        logger.info(f"应用阈值{threshold}后，保留了 {len(valid_tags)} 个标签")
    else:
        valid_tags = set(selected_tags_df['name'].astype(str).unique())
        
    # 3. 读取所有图片的标签
    all_image_tags = {}
    for _, txt_path in file_pairs:
        with open(txt_path, 'r', encoding='utf-8') as f:
            tags = f.read().strip().split(tag_separator)
            tags = [tag.strip() for tag in tags if tag.strip()]
            # 只保留在有效标签集中的标签
            tags = [tag for tag in tags if tag in valid_tags]
            all_image_tags[txt_path] = tags
    
    # 4. 使用与dataset.py相同的逻辑划分数据
    random.seed(42)  # 使用固定种子确保结果可复现
    shuffled_pairs = list(file_pairs)
    random.shuffle(shuffled_pairs)
    
    train_size = int(len(shuffled_pairs) * train_ratio)
    train_pairs = shuffled_pairs[:train_size]
    val_pairs = shuffled_pairs[train_size:]
    
    # 5. 计算训练集和验证集中各标签的出现次数
    train_tag_counter = Counter()
    for _, txt_path in train_pairs:
        tags = all_image_tags.get(txt_path, [])
        train_tag_counter.update(tags)
    
    val_tag_counter = Counter()
    for _, txt_path in val_pairs:
        tags = all_image_tags.get(txt_path, [])
        val_tag_counter.update(tags)
    
    # 6. 分析每个图片的标签数量
    train_tags_per_image = [len(all_image_tags.get(txt_path, [])) for _, txt_path in train_pairs]
    val_tags_per_image = [len(all_image_tags.get(txt_path, [])) for _, txt_path in val_pairs]
    
    train_avg_tags = sum(train_tags_per_image) / len(train_tags_per_image) if train_tags_per_image else 0
    val_avg_tags = sum(val_tags_per_image) / len(val_tags_per_image) if val_tags_per_image else 0
    
    # 7. 统计验证集中没有出现的标签
    train_unique_tags = set(train_tag_counter.keys())
    val_unique_tags = set(val_tag_counter.keys())
    
    missing_in_val = train_unique_tags - val_unique_tags
    
    # 8. 打印结果
    logger.info(f"训练集: {len(train_pairs)} 张图片, {len(train_unique_tags)} 个唯一标签")
    logger.info(f"验证集: {len(val_pairs)} 张图片, {len(val_unique_tags)} 个唯一标签")
    logger.info(f"训练集平均每张图片 {train_avg_tags:.2f} 个标签")
    logger.info(f"验证集平均每张图片 {val_avg_tags:.2f} 个标签")
    logger.info(f"验证集中缺失的标签数量: {len(missing_in_val)}")
    
    # 打印前10个高频标签在训练集和验证集中的分布
    top_tags = [tag for tag, _ in train_tag_counter.most_common(10)]
    logger.info("前10个高频标签在训练集和验证集中的分布:")
    for tag in top_tags:
        logger.info(f"标签 '{tag}': 训练集 {train_tag_counter[tag]} 次, 验证集 {val_tag_counter[tag]} 次")
    
    # 9. 验证集中正样本分析
    val_positive_tags = [tag for tag, count in val_tag_counter.items() if count > 0]
    logger.info(f"验证集中有 {len(val_positive_tags)} 个标签有至少一个正样本")
    if val_positive_tags:
        logger.info(f"验证集中出现的前5个标签: {val_positive_tags[:5]}")
    
    return {
        "train_size": len(train_pairs),
        "val_size": len(val_pairs),
        "train_unique_tags": len(train_unique_tags),
        "val_unique_tags": len(val_unique_tags),
        "train_avg_tags": train_avg_tags,
        "val_avg_tags": val_avg_tags,
        "missing_in_val": len(missing_in_val),
        "val_positive_tags": val_positive_tags,
        "train_tag_counter": train_tag_counter,
        "val_tag_counter": val_tag_counter
    }

if __name__ == "__main__":
    results = analyze_dataset(threshold=5)  # 使用与配置文件相同的阈值
    
    # 可以根据结果进一步处理，例如绘制图表等
    logger.info("分析完成")
