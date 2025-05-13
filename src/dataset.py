import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler # 导入 DistributedSampler
from torchvision import transforms
import logging
import random # 导入 random

# 获取一个logger实例
logger = logging.getLogger(__name__) # 通常 logger 会在 train.py 中配置好

class DanbooruDataset(Dataset):
    def __init__(self, config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
        """
        初始化数据集
        Args:
            config: 配置对象
            mode: "train", "val", "test"
            tags_list: 预定义的标签列表 (可选)
            tag_to_idx: 预定义的标签到索引的映射 (可选)
            rank: 当前进程的 rank (用于日志记录和调试)
            world_size: 总进程数
        """
        self.config = config
        self.mode = mode.lower()
        self.image_dir = config.IMAGE_DIR
        self.rank = rank
        self.world_size = world_size

        # 1. 查找所有图像文件及其对应的 .txt 标签文件
        # 这一步所有进程都执行，以确保文件列表一致，但后续划分可能只在 rank 0 执行或依赖于 sampler
        self.all_file_pairs = self._find_file_pairs()
        if not self.all_file_pairs and self.rank == 0: # 只在 rank 0 打印警告
            logger.warning(f"在目录 {self.image_dir} 中没有找到图像-文本文件对 (支持的图像扩展名: {config.IMAGE_EXTENSIONS})")
        
        # 2. 数据划分 (train/val) - 现在由 DistributedSampler 主要负责，但原始的划分逻辑可以保留用于确定总的训练/验证集
        # 使用固定的种子以确保每次运行结果一致，所有 rank 上的 shuffle 结果应该一致
        # 这样 DistributedSampler 才能从同一个 shuffle 后的列表中取样
        # 注意：DistributedSampler 内部会再次根据 rank 和 world_size 进行划分
        
        # 所有进程都执行相同的 shuffle，以确保 DistributedSampler 基于相同的顺序
        random.seed(42) 
        shuffled_total_pairs = list(self.all_file_pairs)
        random.shuffle(shuffled_total_pairs)
        
        num_total_samples = len(shuffled_total_pairs)
        train_size_total = int(num_total_samples * 0.8) # 基于总样本的划分点
        
        if self.mode == "train":
            self.current_mode_file_pairs = shuffled_total_pairs[:train_size_total]
        elif self.mode == "val":
            self.current_mode_file_pairs = shuffled_total_pairs[train_size_total:]
        elif self.mode == "test": 
            self.current_mode_file_pairs = shuffled_total_pairs # Test 模式使用所有数据
        else:
            raise ValueError(f"不支持的模式: {mode}. 请选择 'train', 'val', 或 'test'.")
        
        if self.rank == 0:
            logger.info(f"模式: {self.mode}, 此模式下的总样本数量 (在被 DDP Sampler 划分前): {len(self.current_mode_file_pairs)}")


        # 3. 构建或使用已有的标签词汇表 (所有进程都应有相同的词汇表)
        if tags_list is None or tag_to_idx is None: 
            self.tags_list, self.tag_to_idx = self._build_vocab_from_selected_tags()
            if self.config.NUM_CLASSES == -1 or self.config.NUM_CLASSES != len(self.tags_list):
                 self.config.NUM_CLASSES = len(self.tags_list)
                 if self.rank == 0: logger.info(f"根据词汇表动态设置 NUM_CLASSES 为: {self.config.NUM_CLASSES}")
        else: 
            self.tags_list = tags_list
            self.tag_to_idx = tag_to_idx
            if self.config.NUM_CLASSES != len(self.tags_list):
                if self.rank == 0: logger.warning(f"配置中的 NUM_CLASSES ({self.config.NUM_CLASSES}) 与提供的词汇表大小 ({len(self.tags_list)}) 不匹配，将使用词汇表大小")
                self.config.NUM_CLASSES = len(self.tags_list)
        
        if self.config.NUM_CLASSES == 0 and self.rank == 0:
            logger.warning("警告: 最终词汇表为空 (NUM_CLASSES = 0)，请检查 selected_tags.csv 和过滤条件")

        # 4. 定义图像变换
        self.transform = self._get_transform()

    def _find_file_pairs(self):
        file_pairs = []
        for ext in self.config.IMAGE_EXTENSIONS:
            for img_path in glob.glob(os.path.join(self.image_dir, f"*{ext}")):
                base_filename, _ = os.path.splitext(img_path)
                txt_path = base_filename + ".txt"
                if os.path.exists(txt_path):
                    file_pairs.append((img_path, txt_path))
        return file_pairs

    def _build_vocab_from_selected_tags(self):
        # ... (这部分逻辑与原版相同，确保所有 rank 得到相同的词汇表) ...
        # (日志也只在 rank 0 打印)
        try:
            selected_tags_df = pd.read_csv(self.config.SELECTED_TAGS_CSV)
        except FileNotFoundError:
            if self.rank == 0: logger.error(f"错误: selected_tags.csv 文件未在路径 {self.config.SELECTED_TAGS_CSV} 找到")
            raise
        
        if 'name' not in selected_tags_df.columns:
            if self.rank == 0: logger.error("错误: selected_tags.csv 文件中必须包含 'name' 列")
            raise ValueError("'name' 列是 selected_tags.csv 中的必需项")

        current_tags_df = selected_tags_df[['name']].copy()
        current_tags_df['name'] = current_tags_df['name'].astype(str)
        
        if self.rank == 0: logger.info(f"从 selected_tags.csv 的 'name' 列初始加载 {len(current_tags_df['name'].unique())} 个唯一标签")

        if 'count' in selected_tags_df.columns and \
           self.config.FILTER_TAG_COUNT_THRESHOLD is not None and \
           self.config.FILTER_TAG_COUNT_THRESHOLD > 0:
            if self.rank == 0: logger.info(f"selected_tags.csv 中存在 'count' 列，将使用阈值 {self.config.FILTER_TAG_COUNT_THRESHOLD} 进行过滤")
            selected_tags_df['count'] = pd.to_numeric(selected_tags_df['count'], errors='coerce')
            original_rows = len(selected_tags_df)
            selected_tags_df.dropna(subset=['count'], inplace=True)
            if len(selected_tags_df) < original_rows and self.rank == 0:
                logger.warning(f"由于 'count' 列存在非数值，已从 selected_tags.csv 中移除了 {original_rows - len(selected_tags_df)} 行")
            filtered_df = selected_tags_df[selected_tags_df['count'] >= self.config.FILTER_TAG_COUNT_THRESHOLD]
            final_tags_list = sorted(list(filtered_df['name'].astype(str).unique()))
            if self.rank == 0: logger.info(f"经过 'count' >= {self.config.FILTER_TAG_COUNT_THRESHOLD} 过滤后，剩余 {len(final_tags_list)} 个唯一标签")
        else:
            final_tags_list = sorted(list(current_tags_df['name'].unique()))
            if self.rank == 0: logger.info("未进行基于 'count' 列的过滤")
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(final_tags_list)}
        if not final_tags_list and self.rank == 0:
            logger.warning("警告: 处理 selected_tags.csv 后，词汇表为空")
        if self.rank == 0: logger.info(f"最终构建的词汇表包含 {len(final_tags_list)} 个标签")
        return final_tags_list, tag_to_idx


    def _get_transform(self):
        # ... (与原版相同) ...
        if self.mode == "train":
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
            ])
        else: 
            return transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD),
            ])

    def __len__(self):
        # 返回的是当前模式下，此 Dataset 对象持有的文件对数量
        # DistributedSampler 会基于这个长度进行划分
        return len(self.current_mode_file_pairs)

    def __getitem__(self, idx):
        # idx 是由 DataLoader (可能通过 Sampler) 提供的索引
        img_path, txt_path = self.current_mode_file_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError: # 只在 rank 0 记录错误，避免日志泛滥
            if self.rank == 0: logger.error(f"错误: 图像文件 {img_path} 未找到，将返回一个红色占位图像")
            image = Image.new("RGB", (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color="red")
        except Exception as e:
            if self.rank == 0: logger.error(f"加载图像 {img_path} 时出错: {e}，将返回一个红色占位图像")
            image = Image.new("RGB", (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), color="red")
        
        image = self.transform(image)

        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                tags_str = f.read().strip()
            current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
            current_img_tags = {tag for tag in current_img_tags if tag} 
        except FileNotFoundError:
            # if self.rank == 0: logger.warning(f"警告: 标签文件 {txt_path} 未找到，该图像将没有标签") # 可能过于频繁
            current_img_tags = set()
        except Exception as e:
            # if self.rank == 0: logger.error(f"读取或解析标签文件 {txt_path} 时出错: {e}，该图像将没有标签")
            current_img_tags = set()
        
        target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        if not self.tag_to_idx and self.config.NUM_CLASSES > 0 :
             if self.rank == 0: logger.warning(f"图像 {img_path} 的标签处理跳过，因为 tag_to_idx 为空但 NUM_CLASSES 为 {self.config.NUM_CLASSES}")
        elif self.tag_to_idx: 
            for tag in current_img_tags:
                if tag in self.tag_to_idx:
                    target[self.tag_to_idx[tag]] = 1.0
        
        return image, target

def get_dataloader(config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
    """
    创建并返回 DataLoader。在分布式训练时使用 DistributedSampler。
    """
    dataset = DanbooruDataset(config, mode, tags_list, tag_to_idx, rank, world_size)
    
    final_tags_list = dataset.tags_list
    final_tag_to_idx = dataset.tag_to_idx

    if config.NUM_CLASSES != len(final_tags_list):
        if rank == 0: logger.info(f"DataLoader: 更新 config.NUM_CLASSES 从 {config.NUM_CLASSES} 到 {len(final_tags_list)}")
        config.NUM_CLASSES = len(final_tags_list)

    sampler = None
    shuffle_in_loader = (mode == "train") # 默认情况下，训练时 shuffle

    if world_size > 1: # 如果是分布式训练
        # drop_last 对于训练通常为 True，以保证所有 GPU batch size 一致
        # 对于验证，通常为 False，以评估所有样本
        drop_last_sampler = (mode == "train")
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                     shuffle=(mode == "train"), drop_last=drop_last_sampler)
        shuffle_in_loader = False # Sampler 已经处理了 shuffle

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE, # 这是每个 GPU 的 batch_size
        shuffle=shuffle_in_loader,
        num_workers=max(0, os.cpu_count() // (2 * world_size) if world_size > 0 and os.cpu_count() else 0), # 每个 DDP 进程分配更少的 worker
        pin_memory=True if config.DEVICE != "cpu" else False, # config.DEVICE 此时可能是 rank
        drop_last=(mode == "train"), # DataLoader 的 drop_last，与 Sampler 的 drop_last 配合
        sampler=sampler
    )
    
    return dataloader, final_tags_list, final_tag_to_idx
