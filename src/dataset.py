import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

# 获取一个logger实例
logger = logging.getLogger(__name__)

class DanbooruDataset(Dataset):
    def __init__(self, config, mode="train", tags_list=None, tag_to_idx=None):
        """
        初始化数据集
        Args:
            config: 配置对象
            mode: "train", "val", "test"，用于数据划分和不同的数据增强策略
            tags_list: 预定义的标签列表 (可选, 用于 val/test 模式以确保词汇表一致)
            tag_to_idx: 预定义的标签到索引的映射 (可选)
        """
        self.config = config
        self.mode = mode.lower()
        self.image_dir = config.IMAGE_DIR

        # 1. 查找所有图像文件及其对应的 .txt 标签文件
        self.file_pairs = self._find_file_pairs()
        if not self.file_pairs:
            logger.warning(f"在目录 {self.image_dir} 中没有找到图像-文本文件对 (支持的图像扩展名: {config.IMAGE_EXTENSIONS})")
        
        # 2. 数据划分 - 使用随机打乱来确保标签分布均衡
        import random
        num_samples = len(self.file_pairs)
        # 使用固定的种子以确保每次运行结果一致
        random.seed(42)
        shuffled_pairs = list(self.file_pairs)
        random.shuffle(shuffled_pairs)
        
        train_size = int(num_samples * 0.8)
        if self.mode == "train":
            self.file_pairs = shuffled_pairs[:train_size]
        elif self.mode == "val":
            self.file_pairs = shuffled_pairs[train_size:]
        elif self.mode == "test": # "test" 模式使用所有找到的文件对
            pass # 使用全部 self.file_pairs
        else:
            raise ValueError(f"不支持的模式: {mode}. 请选择 'train', 'val', 或 'test'.")
        
        logger.info(f"模式: {self.mode}, 样本数量: {len(self.file_pairs)}")

        # 3. 构建或使用已有的标签词汇表
        if tags_list is None or tag_to_idx is None: # 通常在训练模式下首次构建
            self.tags_list, self.tag_to_idx = self._build_vocab_from_selected_tags()
            # 动态更新配置中的类别数 (重要)
            if self.config.NUM_CLASSES == -1 or self.config.NUM_CLASSES != len(self.tags_list):
                 self.config.NUM_CLASSES = len(self.tags_list)
                 logger.info(f"根据词汇表动态设置 NUM_CLASSES 为: {self.config.NUM_CLASSES}")
        else: # 在 val/test 模式或从检查点恢复时，使用传入的词汇表
            self.tags_list = tags_list
            self.tag_to_idx = tag_to_idx
            # 确保配置中的NUM_CLASSES与传入的词汇表大小一致
            if self.config.NUM_CLASSES != len(self.tags_list):
                logger.warning(f"配置中的 NUM_CLASSES ({self.config.NUM_CLASSES}) 与提供的词汇表大小 ({len(self.tags_list)}) 不匹配，将使用词汇表大小")
                self.config.NUM_CLASSES = len(self.tags_list)
        
        if self.config.NUM_CLASSES == 0:
            logger.warning("警告: 最终词汇表为空 (NUM_CLASSES = 0)，请检查 selected_tags.csv 和过滤条件")


        # 4. 定义图像变换
        self.transform = self._get_transform()

    def _find_file_pairs(self):
        """扫描 IMAGE_DIR 目录，查找图像文件及其同名的 .txt 标签文件"""
        file_pairs = []
        for ext in self.config.IMAGE_EXTENSIONS:
            # 使用 glob 查找所有指定扩展名的图像文件
            for img_path in glob.glob(os.path.join(self.image_dir, f"*{ext}")):
                base_filename, _ = os.path.splitext(img_path)
                txt_path = base_filename + ".txt"
                
                if os.path.exists(txt_path):
                    file_pairs.append((img_path, txt_path))
                else:
                    logger.debug(f"图像 {img_path} 对应的标签文件 {txt_path} 未找到，已跳过")
        return file_pairs

    def _build_vocab_from_selected_tags(self):
        """
        从 selected_tags.csv 构建标签词汇表
        会根据 config.FILTER_TAG_COUNT_THRESHOLD (如果 'count' 列存在) 进行过滤
        """
        try:
            selected_tags_df = pd.read_csv(self.config.SELECTED_TAGS_CSV)
        except FileNotFoundError:
            logger.error(f"错误: selected_tags.csv 文件未在路径 {self.config.SELECTED_TAGS_CSV} 找到")
            raise
        
        if 'name' not in selected_tags_df.columns:
            logger.error("错误: selected_tags.csv 文件中必须包含 'name' 列")
            raise ValueError("'name' 列是 selected_tags.csv 中的必需项")

        # 初始标签集来自 'name' 列
        current_tags_df = selected_tags_df[['name']].copy()
        current_tags_df['name'] = current_tags_df['name'].astype(str) # 确保是字符串类型
        
        logger.info(f"从 selected_tags.csv 的 'name' 列初始加载 {len(current_tags_df['name'].unique())} 个唯一标签")

        # 如果存在 'count' 列且阈值有效，则进行过滤
        if 'count' in selected_tags_df.columns and \
           self.config.FILTER_TAG_COUNT_THRESHOLD is not None and \
           self.config.FILTER_TAG_COUNT_THRESHOLD > 0:
            
            logger.info(f"selected_tags.csv 中存在 'count' 列，将使用阈值 {self.config.FILTER_TAG_COUNT_THRESHOLD} 进行过滤")
            # 确保 'count' 列是数字类型，无法转换的设为 NaN
            selected_tags_df['count'] = pd.to_numeric(selected_tags_df['count'], errors='coerce')
            
            # 记录因类型转换失败而被移除的行数
            original_rows = len(selected_tags_df)
            selected_tags_df.dropna(subset=['count'], inplace=True) # 移除 'count' 为 NaN 的行
            if len(selected_tags_df) < original_rows:
                logger.warning(f"由于 'count' 列存在非数值，已从 selected_tags.csv 中移除了 {original_rows - len(selected_tags_df)} 行")

            # 应用阈值过滤
            filtered_df = selected_tags_df[selected_tags_df['count'] >= self.config.FILTER_TAG_COUNT_THRESHOLD]
            final_tags_list = sorted(list(filtered_df['name'].astype(str).unique()))
            logger.info(f"经过 'count' >= {self.config.FILTER_TAG_COUNT_THRESHOLD} 过滤后，剩余 {len(final_tags_list)} 个唯一标签")
        else:
            final_tags_list = sorted(list(current_tags_df['name'].unique()))
            logger.info("未进行基于 'count' 列的过滤 (可能 'count' 列不存在，或阈值未设置/无效)")
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(final_tags_list)}
        
        if not final_tags_list:
            logger.warning("警告: 处理 selected_tags.csv 后，词汇表为空，请检查文件内容和过滤条件")
            # 根据需要，如果词汇表为空，可能需要抛出错误
            # raise ValueError("词汇表不能为空")

        logger.info(f"最终构建的词汇表包含 {len(final_tags_list)} 个标签")
        return final_tags_list, tag_to_idx

    def _get_transform(self):
        """根据模式 (train/val/test) 获取图像变换"""
        if self.mode == "train":
            # 训练模式下使用数据增强
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                # transforms.TrivialAugmentWide(), # 可以考虑更高级的自动增强策略
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
            ])
        else: #验证和测试模式下，通常只进行必要的尺寸调整和归一化
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
            ])

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """获取指定索引的样本 (图像和对应的多标签向量)"""
        img_path, txt_path = self.file_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"错误: 图像文件 {img_path} 未找到，将返回一个红色占位图像")
            image = Image.new("RGB", (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color="red")
        except Exception as e:
            logger.error(f"加载图像 {img_path} 时出错: {e}，将返回一个红色占位图像")
            image = Image.new("RGB", (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color="red")
        
        image = self.transform(image)

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                tags_str = f.read().strip()
            # 使用配置中定义的分隔符分割标签字符串
            current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
            # 移除空字符串标签 (如果分隔符导致)
            current_img_tags = {tag for tag in current_img_tags if tag} 
        except FileNotFoundError:
            logger.warning(f"警告: 标签文件 {txt_path} 未找到，该图像将没有标签")
            current_img_tags = set()
        except Exception as e:
            logger.error(f"读取或解析标签文件 {txt_path} 时出错: {e}，该图像将没有标签")
            current_img_tags = set()
        
        # 创建多标签二元向量 (multi-hot encoding)
        target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        if not self.tag_to_idx and self.config.NUM_CLASSES > 0 : # 词汇表为空但NUM_CLASSES > 0 (不应发生)
             logger.warning(f"图像 {img_path} 的标签处理跳过，因为 tag_to_idx 为空但 NUM_CLASSES 为 {self.config.NUM_CLASSES}")
        elif self.tag_to_idx: # 正常处理
            for tag in current_img_tags:
                if tag in self.tag_to_idx: # 只处理词汇表中存在的标签
                    target[self.tag_to_idx[tag]] = 1.0
        
        return image, target

def get_dataloader(config, mode="train", tags_list=None, tag_to_idx=None):
    """
    创建并返回 DataLoader
    Args:
        config: 配置对象
        mode: "train", "val", "test"
        tags_list, tag_to_idx: 用于 val/test 模式，以确保与训练时词汇表一致
    Returns:
        DataLoader, 最终使用的 tags_list, 最终使用的 tag_to_idx
    """
    dataset = DanbooruDataset(config, mode, tags_list, tag_to_idx)
    
    # 如果在训练模式下 dataset 内部构建了词汇表，则使用该词汇表
    # 否则 (val/test 或 resume 时)，使用传入的词汇表
    final_tags_list = dataset.tags_list if tags_list is None else tags_list
    final_tag_to_idx = dataset.tag_to_idx if tag_to_idx is None else tag_to_idx

    # 确保 NUM_CLASSES 与最终词汇表大小一致
    if config.NUM_CLASSES != len(final_tags_list):
        logger.info(f"DataLoader: 更新 config.NUM_CLASSES 从 {config.NUM_CLASSES} 到 {len(final_tags_list)}")
        config.NUM_CLASSES = len(final_tags_list)

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"), # 训练时打乱数据
        num_workers=max(0, os.cpu_count() // 2 if os.cpu_count() else 0), # 根据CPU核心数设置，0表示在主进程加载
        pin_memory=True if config.DEVICE == "cuda" else False, # 如果使用GPU，启用pin_memory可以加速数据传输
        drop_last=(mode == "train") # 训练时，如果最后一个批次不完整，则丢弃，以保证批次大小一致性
    )
    
    return dataloader, final_tags_list, final_tag_to_idx
