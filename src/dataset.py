import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging
import random # 用于随机划分的回退或测试模式
import numpy as np
# 尝试导入 scikit-multilearn 用于分层划分
try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False
    logging.warning("scikit-multilearn 未安装或导入失败。多标签分层划分将不可用，将回退到随机划分。请运行: pip install scikit-multilearn")

# 获取一个logger实例
logger = logging.getLogger(__name__)

class DanbooruDataset(Dataset):
    def __init__(self, config, mode="train", tags_list=None, tag_to_idx=None):
        """
        初始化数据集 (单卡版本，支持分层划分)
        Args:
            config: 配置对象
            mode: "train", "val", "test"，用于数据划分和不同的数据增强策略
            tags_list: 预定义的标签列表 (可选, 用于 val/test 模式以确保词汇表一致)
            tag_to_idx: 预定义的标签到索引的映射 (可选)
        """
        self.config = config
        self.mode = mode.lower()
        self.image_dir = config.IMAGE_DIR
        self.file_pairs = [] # 初始化为空列表

        # 1. 构建或使用已有的标签词汇表
        # 这一步需要先于数据加载和划分，因为我们需要基于最终的词汇表来创建标签向量
        if tags_list is None or tag_to_idx is None: 
            self.tags_list, self.tag_to_idx = self._build_vocab_from_selected_tags()
            if self.config.NUM_CLASSES == -1 or self.config.NUM_CLASSES != len(self.tags_list):
                 self.config.NUM_CLASSES = len(self.tags_list)
                 logger.info(f"根据词汇表动态设置 NUM_CLASSES 为: {self.config.NUM_CLASSES}")
        else: 
            self.tags_list = tags_list
            self.tag_to_idx = tag_to_idx
            if self.config.NUM_CLASSES != len(self.tags_list):
                logger.warning(f"配置中的 NUM_CLASSES ({self.config.NUM_CLASSES}) 与提供的词汇表大小 ({len(self.tags_list)}) 不匹配，将使用词汇表大小")
                self.config.NUM_CLASSES = len(self.tags_list)
        
        if self.config.NUM_CLASSES == 0:
            logger.warning("警告: 最终词汇表为空 (NUM_CLASSES = 0)，请检查 selected_tags.csv 和过滤条件")
            self.transform = self._get_transform() # 即使为空也定义transform
            return # 如果没有类别，则数据集为空

        # 2. 查找所有图像文件路径
        all_image_paths = self._find_image_paths()

        if not all_image_paths:
            logger.warning(f"在目录 {self.image_dir} 中没有找到图像文件 (支持的图像扩展名: {config.IMAGE_EXTENSIONS})")
            self.transform = self._get_transform()
            return

        # 3. 为所有有效的图像-标签对创建用于划分的数据
        X_filepaths_for_split = [] # 存储有效的文件路径 (图像和对应txt都存在)
        y_labels_list_for_split = [] # 存储对应的多标签向量列表

        logger.info("正在为所有图像准备标签向量以进行数据划分...")
        # 对于单卡，不需要 disable tqdm
        temp_progress_bar = tqdm(all_image_paths, desc="读取标签文件进行划分准备")
        for img_path in temp_progress_bar:
            base_filename, _ = os.path.splitext(img_path)
            txt_path = base_filename + ".txt"
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        tags_str = f.read().strip()
                    current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
                    current_img_tags = {tag for tag in current_img_tags if tag}
                    
                    # 创建标签向量，即使 NUM_CLASSES 为0或tag_to_idx为空，也尝试创建空向量
                    label_vector = torch.zeros(self.config.NUM_CLASSES, dtype=torch.int8) # 使用 int8 节省内存
                    if self.tag_to_idx and self.config.NUM_CLASSES > 0:
                        for tag in current_img_tags:
                            if tag in self.tag_to_idx:
                                label_vector[self.tag_to_idx[tag]] = 1
                    
                    X_filepaths_for_split.append(img_path)
                    y_labels_list_for_split.append(label_vector.numpy())
                except Exception as e:
                    logger.warning(f"读取或处理标签文件 {txt_path} 出错: {e}，跳过图像 {img_path}")
        
        if not X_filepaths_for_split:
            logger.error("没有找到任何有效的图像-标签对用于数据划分。数据集将为空。")
            self.transform = self._get_transform()
            return

        X_filepaths_np = np.array(X_filepaths_for_split)
        y_labels_np = np.array(y_labels_list_for_split)

        # 4. 数据划分 (train/val)
        if self.mode == "train" or self.mode == "val":
            val_split_ratio = getattr(self.config, 'VALIDATION_SPLIT_RATIO', 0.2)
            if not (0 < val_split_ratio < 1):
                logger.warning(f"无效的 VALIDATION_SPLIT_RATIO: {val_split_ratio}，将使用默认值 0.2")
                val_split_ratio = 0.2

            # 确保 y_labels_np 至少有两维，即使只有一个标签
            if y_labels_np.ndim == 1 and self.config.NUM_CLASSES == 1:
                y_labels_np = y_labels_np.reshape(-1, 1)


            if SKMULTILEARN_AVAILABLE and y_labels_np.shape[0] > 1 and y_labels_np.shape[1] > 0:
                logger.info(f"使用 IterativeStratification 进行数据划分，验证集比例: {val_split_ratio}")
                # IterativeStratification 的 order 参数可以影响划分，通常默认为1或2。
                # sample_distribution_per_fold: 第一个元素是验证集比例，第二个是训练集比例
                stratifier = IterativeStratification(n_splits=2, order=1, 
                                                     sample_distribution_per_fold=[val_split_ratio, 1.0 - val_split_ratio])
                try:
                    # stratifier.split 返回 (train_indices, test_indices) 的生成器
                    # 由于 n_splits=2，它只会产生一对。test_indices 对应第一个比例 (val_split_ratio)
                    # 所以，第一个是验证集索引，第二个是训练集索引
                    val_indices, train_indices = next(stratifier.split(X_filepaths_np, y_labels_np))
                    logger.info(f"分层划分完成：训练集样本数 {len(train_indices)}, 验证集样本数 {len(val_indices)}")
                except ValueError as e: # IterativeStratification 可能因数据特性失败 (例如某些标签组合只出现一次)
                    logger.error(f"IterativeStratification 执行失败: {e}。将回退到随机划分。")
                    # 回退到随机划分
                    indices = np.arange(len(X_filepaths_np))
                    np.random.seed(42) # 使用固定种子确保一致性
                    np.random.shuffle(indices)
                    split_point = int(len(indices) * (1.0 - val_split_ratio))
                    train_indices = indices[:split_point]
                    val_indices = indices[split_point:]
            else:
                if not SKMULTILEARN_AVAILABLE:
                    logger.info(f"scikit-multilearn 不可用，回退到随机划分。验证集比例: {val_split_ratio}")
                else:
                    logger.info(f"数据不适合 IterativeStratification (样本数: {y_labels_np.shape[0]}, 标签数: {y_labels_np.shape[1]})，回退到随机划分。验证集比例: {val_split_ratio}")
                
                indices = np.arange(len(X_filepaths_np))
                np.random.seed(42) # 确保随机划分的一致性
                np.random.shuffle(indices)
                split_point = int(len(indices) * (1.0 - val_split_ratio)) # 第一个是训练集
                train_indices = indices[:split_point]
                val_indices = indices[split_point:]

            if self.mode == "train":
                selected_indices = train_indices
            else: # self.mode == "val"
                selected_indices = val_indices
            
            for idx in selected_indices:
                img_path = X_filepaths_np[idx]
                base, _ = os.path.splitext(img_path)
                txt_path = base + ".txt"
                self.file_pairs.append((img_path, txt_path))

        elif self.mode == "test":
            logger.info("测试模式：使用所有找到的有效图像-标签对。")
            # 对于测试模式，我们使用所有找到的 X_filepaths_for_split
            for img_path_test in X_filepaths_for_split:
                base_test, _ = os.path.splitext(img_path_test)
                txt_path_test = base_test + ".txt"
                self.file_pairs.append((img_path_test, txt_path_test))
        else:
            raise ValueError(f"不支持的模式: {self.mode}. 请选择 'train', 'val', 或 'test'.")
        
        logger.info(f"模式: {self.mode}, 此模式下的样本数量: {len(self.file_pairs)}")
        if len(self.file_pairs) == 0:
             logger.error(f"警告: 模式 '{self.mode}' 数据集为空，请检查数据源和划分逻辑！")

        # 5. 定义图像变换
        self.transform = self._get_transform()

    def _find_image_paths(self):
        """扫描 IMAGE_DIR 目录，仅查找图像文件路径"""
        image_paths = []
        for ext in self.config.IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        # 对找到的路径进行排序，确保每次加载顺序一致，这对于后续的划分很重要
        image_paths.sort() 
        return image_paths

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

        current_tags_df = selected_tags_df[['name']].copy()
        current_tags_df['name'] = current_tags_df['name'].astype(str)
        
        logger.info(f"从 selected_tags.csv 的 'name' 列初始加载 {len(current_tags_df['name'].unique())} 个唯一标签")

        filter_threshold = getattr(self.config, 'FILTER_TAG_COUNT_THRESHOLD', 0)

        if 'count' in selected_tags_df.columns and \
           filter_threshold is not None and \
           filter_threshold > 0:
            
            logger.info(f"selected_tags.csv 中存在 'count' 列，将使用阈值 {filter_threshold} 进行过滤")
            selected_tags_df['count'] = pd.to_numeric(selected_tags_df['count'], errors='coerce')
            
            original_rows = len(selected_tags_df)
            selected_tags_df.dropna(subset=['count'], inplace=True) 
            if len(selected_tags_df) < original_rows:
                logger.warning(f"由于 'count' 列存在非数值，已从 selected_tags.csv 中移除了 {original_rows - len(selected_tags_df)} 行")

            filtered_df = selected_tags_df[selected_tags_df['count'] >= filter_threshold]
            final_tags_list = sorted(list(filtered_df['name'].astype(str).unique()))
            logger.info(f"经过 'count' >= {filter_threshold} 过滤后，剩余 {len(final_tags_list)} 个唯一标签")
        else:
            final_tags_list = sorted(list(current_tags_df['name'].unique()))
            logger.info("未进行基于 'count' 列的过滤 (可能 'count' 列不存在，或阈值未设置/无效)")
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(final_tags_list)}
        
        if not final_tags_list:
            logger.warning("警告: 处理 selected_tags.csv 后，词汇表为空，请检查文件内容和过滤条件")

        logger.info(f"最终构建的词汇表包含 {len(final_tags_list)} 个标签")
        return final_tags_list, tag_to_idx

    def _get_transform(self):
        """根据模式 (train/val/test) 获取图像变换"""
        img_size = getattr(self.config, 'IMAGE_SIZE', 224)
        norm_mean = getattr(self.config, 'NORM_MEAN', [0.485, 0.456, 0.406])
        norm_std = getattr(self.config, 'NORM_STD', [0.229, 0.224, 0.225])

        if self.mode == "train":
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ])
        else: #验证和测试模式下
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ])

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.file_pairs)

    def __getitem__(self, idx):
        """获取指定索引的样本 (图像和对应的多标签向量)"""
        if idx >= len(self.file_pairs): # 索引越界检查
            # 这种情况理论上不应由 DataLoader 触发，但作为防御性编程
            raise IndexError(f"索引 {idx} 超出数据集范围 {len(self.file_pairs)}")

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
            current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
            current_img_tags = {tag for tag in current_img_tags if tag} 
        except FileNotFoundError:
            # 在 __init__ 中已经过滤了没有txt的图像，理论上不应发生，除非文件在之后被删除
            logger.warning(f"警告: 标签文件 {txt_path} 在 __getitem__ 中未找到，该图像将没有标签")
            current_img_tags = set()
        except Exception as e:
            logger.error(f"读取或解析标签文件 {txt_path} 时出错: {e}，该图像将没有标签")
            current_img_tags = set()
        
        target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32)
        if not self.tag_to_idx and self.config.NUM_CLASSES > 0 : 
             logger.warning(f"图像 {img_path} 的标签处理跳过，因为 tag_to_idx 为空但 NUM_CLASSES 为 {self.config.NUM_CLASSES}")
        elif self.tag_to_idx and self.config.NUM_CLASSES > 0: 
            for tag in current_img_tags:
                if tag in self.tag_to_idx: 
                    target[self.tag_to_idx[tag]] = 1.0
        
        return image, target

def get_dataloader(config, mode="train", tags_list=None, tag_to_idx=None):
    """
    创建并返回 DataLoader (单卡版本)
    Args:
        config: 配置对象
        mode: "train", "val", "test"
        tags_list, tag_to_idx: 用于 val/test 模式，以确保与训练时词汇表一致
    Returns:
        DataLoader, 最终使用的 tags_list, 最终使用的 tag_to_idx
    """
    dataset = DanbooruDataset(config, mode, tags_list, tag_to_idx)
    
    if len(dataset) == 0:
        logger.warning(f"模式 '{mode}' 的数据集为空，返回一个空的 DataLoader。")
        # 创建一个可以迭代但实际上是空的 DataLoader
        # 注意：如果 batch_size > 0 而数据集为空，DataLoader 会在尝试获取第一个 batch 时出错
        # 如果 dataset 为空，sampler 设为 None
        return DataLoader(dataset, batch_size=max(1, config.BATCH_SIZE), shuffle=False, num_workers=0), dataset.tags_list, dataset.tag_to_idx

    final_tags_list = dataset.tags_list
    final_tag_to_idx = dataset.tag_to_idx

    if config.NUM_CLASSES != len(final_tags_list):
        logger.info(f"DataLoader: 更新 config.NUM_CLASSES 从 {config.NUM_CLASSES} 到 {len(final_tags_list)}")
        config.NUM_CLASSES = len(final_tags_list)

    # 单卡训练，不使用 DistributedSampler
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == "train"), # 训练时打乱数据
        num_workers=max(0, os.cpu_count() // 2 if os.cpu_count() else 0), 
        pin_memory=True if config.DEVICE == "cuda" else False, 
        drop_last=(mode == "train") 
    )
    
    return dataloader, final_tags_list, final_tag_to_idx
