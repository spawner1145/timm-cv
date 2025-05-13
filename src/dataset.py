import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import logging
import random # 仍然保留 random 用于其他可能的随机操作或作为备选
import numpy as np
from sklearn.model_selection import train_test_split # 用于单标签或简单场景的备选
import tqdm
try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False
    logging.warning("scikit-multilearn 未安装或导入失败。多标签分层划分将不可用，将回退到随机划分。请运行: pip install scikit-multilearn")

# 获取一个logger实例
logger = logging.getLogger(__name__)

class DanbooruDataset(Dataset):
    def __init__(self, config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
        """
        初始化数据集
        Args:
            config: 配置对象
            mode: "train", "val", "test"
            tags_list: 预定义的标签列表 (可选)
            tag_to_idx: 预定义的标签到索引的映射 (可选)
            rank: 当前进程的 rank
            world_size: 总进程数
        """
        self.config = config
        self.mode = mode.lower()
        self.image_dir = config.IMAGE_DIR
        self.rank = rank
        self.world_size = world_size

        # 1. 构建或使用已有的标签词汇表 (所有进程都应有相同的词汇表)
        # 这一步需要先于数据加载和划分，因为我们需要基于最终的词汇表来创建标签向量
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
        
        if self.config.NUM_CLASSES == 0:
            if self.rank == 0: logger.warning("警告: 最终词汇表为空 (NUM_CLASSES = 0)，请检查 selected_tags.csv 和过滤条件")
            # 如果词汇表为空，后续处理会出问题，可以提前抛出错误或返回空数据集
            self.current_mode_file_pairs = []
            self.transform = self._get_transform() # 即使为空也定义transform
            return


        # 2. 查找所有图像文件及其对应的 .txt 标签文件
        all_file_paths = self._find_image_paths() # 只获取图像路径列表

        if not all_file_paths:
            if self.rank == 0: 
                logger.warning(f"在目录 {self.image_dir} 中没有找到图像文件 (支持的图像扩展名: {config.IMAGE_EXTENSIONS})")
            self.current_mode_file_pairs = []
            self.transform = self._get_transform()
            return
            
        # 3. 为所有图像创建标签向量 (multi-hot encoding) 以用于分层划分
        # 这一步可能会比较耗时，因为它需要读取所有图像的 .txt 文件
        # 只有在 mode 不是 "test" 且 skmultilearn 可用时才进行分层划分的准备
        # "test" 模式通常使用所有数据，不需要划分
        
        X_filepaths = [] # 存储有效的文件路径 (图像和对应txt都存在)
        y_labels_list = [] # 存储对应的多标签向量列表

        if self.mode != "test": # 测试集通常不参与划分，或者有单独的测试集文件列表
            if self.rank == 0: logger.info("正在为所有图像准备标签向量以进行数据划分...")
            temp_progress_bar = tqdm(all_file_paths, desc="读取标签文件", disable=(self.rank != 0))
            for img_path in temp_progress_bar:
                base_filename, _ = os.path.splitext(img_path)
                txt_path = base_filename + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            tags_str = f.read().strip()
                        current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
                        current_img_tags = {tag for tag in current_img_tags if tag}
                        
                        label_vector = torch.zeros(self.config.NUM_CLASSES, dtype=torch.int8) # 使用 int8 节省内存
                        for tag in current_img_tags:
                            if tag in self.tag_to_idx:
                                label_vector[self.tag_to_idx[tag]] = 1
                        
                        X_filepaths.append(img_path) # 只保留有对应txt文件的图像
                        y_labels_list.append(label_vector.numpy()) # 转换为 NumPy array
                    except Exception as e:
                        if self.rank == 0: logger.warning(f"读取或处理标签文件 {txt_path} 出错: {e}，跳过图像 {img_path}")
                # else:
                    # if self.rank == 0: logger.debug(f"图像 {img_path} 的标签文件 {txt_path} 未找到，跳过。")
            
            if not X_filepaths:
                if self.rank == 0: logger.error("没有找到任何有效的图像-标签对用于数据划分。")
                self.current_mode_file_pairs = []
                self.transform = self._get_transform()
                return

            y_labels_np = np.array(y_labels_list)
            X_filepaths_np = np.array(X_filepaths) # 将文件路径也转为NumPy数组，方便索引

            # 4. 数据划分
            # 使用 IterativeStratification (如果可用) 或回退到随机划分
            # train_ratio 通常是 0.8，val_ratio 是 0.2
            # 注意: IterativeStratification 需要 X 和 y。X 可以是索引或实际数据。这里用索引。
            # 我们需要的是文件对 (img_path, txt_path)，所以划分后要重新构建
            
            # IterativeStratification 的 train_test_split 行为是划分一次，所以我们需要 (1-val_ratio) 作为 test_size
            # 例如，如果 val_ratio 是 0.2，那么 test_size 应该是 0.2，得到 80% 训练，20% 验证
            # 如果配置文件中没有明确的验证集比例，我们默认使用 0.2
            val_split_ratio = getattr(self.config, 'VALIDATION_SPLIT_RATIO', 0.2)
            if not (0 < val_split_ratio < 1):
                if self.rank == 0: logger.warning(f"无效的 VALIDATION_SPLIT_RATIO: {val_split_ratio}，将使用默认值 0.2")
                val_split_ratio = 0.2

            if SKMULTILEARN_AVAILABLE and y_labels_np.shape[0] > 1 and y_labels_np.shape[1] > 0: # 确保有数据和标签
                if self.rank == 0: logger.info(f"使用 IterativeStratification 进行数据划分，验证集比例: {val_split_ratio}")
                # n_splits 通常是 2，但 IterativeStratification 的 k_fold 行为不同。
                # 我们需要的是一个 train/test split。
                # IterativeStratification 的 order 参数可以影响划分，通常默认为1或2。
                # test_size 参数指定了第二个集合的大小。
                stratifier = IterativeStratification(n_splits=2, order=1, sample_distribution_per_fold=[val_split_ratio, 1.0-val_split_ratio])
                
                # stratifier.split 返回的是 (train_indices, test_indices) 的生成器
                # 由于 n_splits=2，它只会产生一对。
                try:
                    train_indices, val_indices = next(stratifier.split(X_filepaths_np, y_labels_np))
                except Exception as e: # IterativeStratification 可能因数据特性失败
                    if self.rank == 0:
                        logger.error(f"IterativeStratification 执行失败: {e}。将回退到随机划分。")
                    # 回退到随机划分
                    indices = np.arange(len(X_filepaths_np))
                    # 使用固定的随机种子，确保所有进程的划分一致
                    np.random.seed(42)
                    np.random.shuffle(indices)
                    split_point = int(len(indices) * (1.0 - val_split_ratio))
                    train_indices = indices[:split_point]
                    val_indices = indices[split_point:]

            else: # skmultilearn 不可用或数据不适合分层（例如只有一个样本或没有标签维度）
                if self.rank == 0:
                    if not SKMULTILEARN_AVAILABLE:
                        logger.info(f"scikit-multilearn 不可用，回退到随机划分。验证集比例: {val_split_ratio}")
                    else:
                        logger.info(f"数据不适合 IterativeStratification (样本数: {y_labels_np.shape[0]}, 标签数: {y_labels_np.shape[1]})，回退到随机划分。验证集比例: {val_split_ratio}")

                indices = np.arange(len(X_filepaths_np))
                np.random.seed(42) # 确保随机划分的一致性
                np.random.shuffle(indices)
                split_point = int(len(indices) * (1.0 - val_split_ratio))
                train_indices = indices[:split_point]
                val_indices = indices[split_point:]

            if self.mode == "train":
                selected_indices = train_indices
            elif self.mode == "val":
                selected_indices = val_indices
            else: # "test" 模式不应该进入这个分支，但作为保险
                selected_indices = np.arange(len(X_filepaths_np))

            # 从选中的索引构建 self.current_mode_file_pairs
            self.current_mode_file_pairs = []
            for idx in selected_indices:
                img_path = X_filepaths_np[idx]
                base, _ = os.path.splitext(img_path)
                txt_path = base + ".txt"
                self.current_mode_file_pairs.append((img_path, txt_path))
        
        else: # self.mode == "test"
            if self.rank == 0: logger.info("测试模式：使用所有找到的有效图像-标签对。")
            self.current_mode_file_pairs = []
            temp_progress_bar_test = tqdm(all_file_paths, desc="检查测试集文件", disable=(self.rank != 0))
            for img_path in temp_progress_bar_test:
                base_filename, _ = os.path.splitext(img_path)
                txt_path = base_filename + ".txt"
                if os.path.exists(txt_path): # 测试集也需要标签文件（即使只是用于格式一致性）
                    self.current_mode_file_pairs.append((img_path, txt_path))
            if not self.current_mode_file_pairs and self.rank == 0:
                logger.warning("测试模式下未找到任何有效的图像-标签对。")


        if self.rank == 0:
            logger.info(f"模式: {self.mode}, 此模式下的样本数量 (划分后，被 DDP Sampler 处理前): {len(self.current_mode_file_pairs)}")
            if self.mode != "test" and len(self.current_mode_file_pairs) == 0:
                 logger.error(f"警告: 模式 '{self.mode}' 数据集为空，请检查数据源和划分逻辑！")


        # 5. 定义图像变换
        self.transform = self._get_transform()

    def _find_image_paths(self):
        """扫描 IMAGE_DIR 目录，仅查找图像文件路径"""
        image_paths = []
        for ext in self.config.IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        return image_paths

    def _build_vocab_from_selected_tags(self):
        """
        从 selected_tags.csv 构建标签词汇表 (与原版逻辑相同)
        """
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

        filter_threshold = getattr(self.config, 'FILTER_TAG_COUNT_THRESHOLD', 0) # 从配置获取阈值

        if 'count' in selected_tags_df.columns and \
           filter_threshold is not None and \
           filter_threshold > 0:
            
            if self.rank == 0: logger.info(f"selected_tags.csv 中存在 'count' 列，将使用阈值 {filter_threshold} 进行过滤")
            selected_tags_df['count'] = pd.to_numeric(selected_tags_df['count'], errors='coerce')
            
            original_rows = len(selected_tags_df)
            selected_tags_df.dropna(subset=['count'], inplace=True) 
            if len(selected_tags_df) < original_rows and self.rank == 0:
                logger.warning(f"由于 'count' 列存在非数值，已从 selected_tags.csv 中移除了 {original_rows - len(selected_tags_df)} 行")

            filtered_df = selected_tags_df[selected_tags_df['count'] >= filter_threshold]
            final_tags_list = sorted(list(filtered_df['name'].astype(str).unique()))
            if self.rank == 0: logger.info(f"经过 'count' >= {filter_threshold} 过滤后，剩余 {len(final_tags_list)} 个唯一标签")
        else:
            final_tags_list = sorted(list(current_tags_df['name'].unique()))
            if self.rank == 0: logger.info("未进行基于 'count' 列的过滤 (可能 'count' 列不存在，或阈值未设置/无效)")
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(final_tags_list)}
        
        if not final_tags_list and self.rank == 0:
            logger.warning("警告: 处理 selected_tags.csv 后，词汇表为空，请检查文件内容和过滤条件")

        if self.rank == 0: logger.info(f"最终构建的词汇表包含 {len(final_tags_list)} 个标签")
        return final_tags_list, tag_to_idx

    def _get_transform(self):
        """根据模式 (train/val/test) 获取图像变换 (与原版逻辑相同)"""
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
        else: 
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std),
            ])

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.current_mode_file_pairs)

    def __getitem__(self, idx):
        """获取指定索引的样本 (图像和对应的多标签向量)"""
        if idx >= len(self.current_mode_file_pairs): # 索引越界检查
            raise IndexError(f"索引 {idx} 超出数据集范围 {len(self.current_mode_file_pairs)}")

        img_path, txt_path = self.current_mode_file_pairs[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
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
            # if self.rank == 0: logger.warning(f"警告: 标签文件 {txt_path} 未找到，该图像将没有标签")
            current_img_tags = set() # 在 __init__ 中已经过滤了没有txt的图像，理论上不应发生
        except Exception as e:
            # if self.rank == 0: logger.error(f"读取或解析标签文件 {txt_path} 时出错: {e}，该图像将没有标签")
            current_img_tags = set()
        
        target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32) # 损失函数通常期望 float32
        if not self.tag_to_idx and self.config.NUM_CLASSES > 0 : 
             if self.rank == 0: logger.warning(f"图像 {img_path} 的标签处理跳过，因为 tag_to_idx 为空但 NUM_CLASSES 为 {self.config.NUM_CLASSES}")
        elif self.tag_to_idx: 
            for tag in current_img_tags:
                if tag in self.tag_to_idx: 
                    target[self.tag_to_idx[tag]] = 1.0
        
        return image, target

def get_dataloader(config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
    """
    创建并返回 DataLoader (与原版多卡逻辑相同)
    """
    dataset = DanbooruDataset(config, mode, tags_list, tag_to_idx, rank, world_size)
    
    # 如果数据集为空 (例如，在划分后某个模式下没有样本了)
    if len(dataset) == 0:
        if rank == 0:
            logger.warning(f"模式 '{mode}' 的数据集为空，返回一个空的 DataLoader。")
        # 返回一个可以迭代但实际上是空的 DataLoader，或者根据需要抛出错误
        # 这里我们返回一个能处理空数据集的 DataLoader
        # 注意：如果 batch_size > 0 而数据集为空，DataLoader 会在尝试获取第一个 batch 时出错
        # 因此，如果数据集为空，我们可能需要特殊处理或确保调用者能处理这种情况
        # 一个简单的方法是，如果数据集为空，则 batch_size 设为 1 (或任何正数)，但迭代器会立即结束
        # 或者，如果确实不希望空的dataloader，这里可以 raise error
        # return None, dataset.tags_list, dataset.tag_to_idx # 或者返回 None
        # 为了能继续运行，我们还是创建一个 DataLoader，但它会是空的
        # 如果 dataset 为空，DistributedSampler(dataset,...) 会出错，所以 sampler 设为 None
        sampler = None
        shuffle_in_loader = False # 空数据集不需要shuffle
        actual_batch_size = config.BATCH_SIZE if config.BATCH_SIZE > 0 else 1
    else:
        final_tags_list = dataset.tags_list
        final_tag_to_idx = dataset.tag_to_idx

        if config.NUM_CLASSES != len(final_tags_list):
            if rank == 0: logger.info(f"DataLoader: 更新 config.NUM_CLASSES 从 {config.NUM_CLASSES} 到 {len(final_tags_list)}")
            config.NUM_CLASSES = len(final_tags_list)

        sampler = None
        shuffle_in_loader = (mode == "train") 

        if world_size > 1: 
            drop_last_sampler = (mode == "train")
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                         shuffle=(mode == "train"), drop_last=drop_last_sampler)
            shuffle_in_loader = False 
        actual_batch_size = config.BATCH_SIZE


    dataloader = DataLoader(
        dataset,
        batch_size=max(1, actual_batch_size), # 确保 batch_size 至少为1
        shuffle=shuffle_in_loader,
        num_workers=max(0, os.cpu_count() // (2 * max(1,world_size)) if os.cpu_count() else 0),
        pin_memory=True if config.DEVICE != "cpu" and isinstance(config.DEVICE, int) else False, # DEVICE可能是rank(int)或'cpu'
        drop_last=(mode == "train" and len(dataset) > 0), # 只有在训练且数据集非空时才drop_last
        sampler=sampler
    )
    
    return dataloader, dataset.tags_list, dataset.tag_to_idx
