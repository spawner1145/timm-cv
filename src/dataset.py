import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import logging
import random
import numpy as np
from tqdm import tqdm

try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False
    # logging.warning 在 main_worker 中由 rank 0 记录

logger = logging.getLogger(__name__)

# --- 平衡采样策略的硬编码配置 ---
ENABLE_BALANCED_SAMPLING_TRAIN_HARDCODED = True 
MAX_SAMPLES_PER_CLASS_BALANCED_HARDCODED = 1000  
TARGET_TRAIN_SET_SIZE_BALANCED_HARDCODED = -1 
BALANCED_SAMPLING_SEED = 42
# ------------------------------------

class DanbooruDataset(Dataset):
    def __init__(self, config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
        self.config = config
        self.mode = mode.lower()
        self.image_dir = config.IMAGE_DIR
        self.rank = rank
        self.world_size = world_size
        self.file_pairs = [] 

        if tags_list is None or tag_to_idx is None: 
            self.tags_list, self.tag_to_idx = self._build_vocab_from_selected_tags()
            if self.config.NUM_CLASSES == -1 or self.config.NUM_CLASSES != len(self.tags_list):
                 self.config.NUM_CLASSES = len(self.tags_list)
                 if self.rank == 0: logger.info(f"根据词汇表动态设置 NUM_CLASSES 为: {self.config.NUM_CLASSES}")
        else: 
            self.tags_list = tags_list
            self.tag_to_idx = tag_to_idx
            if self.config.NUM_CLASSES != len(self.tags_list):
                if self.rank == 0: logger.warning(f"配置中的 NUM_CLASSES ({self.config.NUM_CLASSES}) 与提供的词汇表大小 ({len(self.tags_list)}) 不匹配")
                self.config.NUM_CLASSES = len(self.tags_list)
        
        if self.config.NUM_CLASSES == 0:
            if self.rank == 0: logger.warning("警告: 最终词汇表为空 (NUM_CLASSES = 0)")
            self.transform = self._get_transform()
            return

        all_image_paths = self._find_image_paths()
        if not all_image_paths:
            if self.rank == 0: logger.warning(f"在目录 {self.image_dir} 中没有找到图像文件")
            self.transform = self._get_transform()
            return
        
        use_balanced_sampling_for_train = ENABLE_BALANCED_SAMPLING_TRAIN_HARDCODED

        if self.mode == "train" and use_balanced_sampling_for_train:
            if self.rank == 0: logger.info("为训练集启用平衡采样策略...")
            # 所有 rank 都需要构建这个映射，以确保后续采样逻辑一致
            tag_idx_to_img_paths, img_path_to_txt_path_map = self._build_tag_to_image_map(all_image_paths)
            if not tag_idx_to_img_paths and not img_path_to_txt_path_map :
                 if self.rank == 0: logger.error("平衡采样预处理失败或未找到有效数据，训练集将为空。")
            else:
                self._apply_balanced_sampling(tag_idx_to_img_paths, img_path_to_txt_path_map)
        else: 
            if self.rank == 0: logger.info(f"为模式 '{self.mode}' 应用标准划分策略 (分层或随机)...")
            # 标准划分也需要在所有rank上执行相同的逻辑，以确保数据集一致
            self._apply_standard_splitting(all_image_paths)
        
        if self.rank == 0: 
            logger.info(f"模式: {self.mode}, 最终样本数量 (rank {self.rank} 在 DDP Sampler 前): {len(self.file_pairs)}")
            if len(self.file_pairs) == 0 and self.mode != "test":
                 logger.error(f"警告: 模式 '{self.mode}' 数据集为空 (rank {self.rank})，请检查配置和数据源！")
        
        self.transform = self._get_transform()

    def _build_tag_to_image_map(self, all_image_paths):
        tag_idx_to_img_paths = [[] for _ in range(self.config.NUM_CLASSES)]
        img_path_to_txt_path = {}
        valid_image_paths_count = 0
        
        prog_desc = "构建标签到图像映射"
        disable_tqdm_preproc = (self.rank != 0 and self.world_size > 1) # 只在rank 0显示tqdm (如果是多卡)
        
        for img_path in tqdm(all_image_paths, desc=prog_desc, disable=disable_tqdm_preproc):
            base_filename, _ = os.path.splitext(img_path)
            txt_path = base_filename + ".txt"
            if os.path.exists(txt_path):
                img_path_to_txt_path[img_path] = txt_path 
                valid_image_paths_count +=1
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f_txt:
                        tags_str = f_txt.read().strip()
                    current_img_tags_in_file = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
                    current_img_tags_in_file = {tag for tag in current_img_tags_in_file if tag}
                    for tag_name in current_img_tags_in_file:
                        if tag_name in self.tag_to_idx:
                            tag_idx = self.tag_to_idx[tag_name]
                            tag_idx_to_img_paths[tag_idx].append(img_path)
                except Exception as e:
                    if self.rank == 0: logger.warning(f"读取标签文件 {txt_path} 出错: {e}")
        
        if valid_image_paths_count == 0 and self.rank == 0 :
             logger.error("在构建标签映射时，没有找到任何带有对应txt文件的有效图像。")
             return {}, {}
             
        return tag_idx_to_img_paths, img_path_to_txt_path

    def _apply_balanced_sampling(self, tag_idx_to_img_paths, img_path_to_txt_path_map):
        """为训练集应用平衡采样策略。所有 rank 执行相同的逻辑以保证 self.file_pairs 一致。"""
        seeded_random = random.Random(BALANCED_SAMPLING_SEED) # 每个rank使用相同的种子
        
        # 对每个类别的图像列表进行加籽随机打乱
        # 注意：这里直接修改了传入的 tag_idx_to_img_paths 列表的内容顺序
        for i in range(len(tag_idx_to_img_paths)):
            seeded_random.shuffle(tag_idx_to_img_paths[i])

        self.file_pairs = [] 
        samples_added_for_class_count = [0] * self.config.NUM_CLASSES
        next_sample_idx_for_class = [0] * self.config.NUM_CLASSES 

        max_samples_per_class = MAX_SAMPLES_PER_CLASS_BALANCED_HARDCODED
        target_total_train_size = TARGET_TRAIN_SET_SIZE_BALANCED_HARDCODED
        
        active_class_indices = [idx for idx, paths in enumerate(tag_idx_to_img_paths) if paths]
        if not active_class_indices:
            if self.rank == 0: logger.warning("平衡采样：没有找到任何带有有效标签的图像。训练集将为空。")
            return

        num_added_total = 0
        safety_break_max_loops = (target_total_train_size * 2 if target_total_train_size > 0 else self.config.NUM_CLASSES * max_samples_per_class * 2)
        if safety_break_max_loops <= 0 : 
            safety_break_max_loops = len(img_path_to_txt_path_map) * 2 if img_path_to_txt_path_map else 100000 
        
        if self.rank == 0: logger.info(f"平衡采样启动：每类最多 {max_samples_per_class} 个样本，目标总大小: {'无限制' if target_total_train_size == -1 else target_total_train_size}")

        loop_iteration = 0
        while loop_iteration < safety_break_max_loops:
            loop_iteration +=1
            if target_total_train_size > 0 and num_added_total >= target_total_train_size:
                if self.rank == 0: logger.info(f"已达到目标训练集大小: {num_added_total}")
                break
            
            if not active_class_indices:
                if self.rank == 0: logger.info("所有活动类别均已耗尽或达到采样上限。")
                break

            made_a_selection_this_round = False
            seeded_random.shuffle(active_class_indices) # 确保类别选择顺序在各rank间一致

            temp_active_classes_next_round = [] 

            for class_idx in active_class_indices:
                can_add_more_for_this_class = samples_added_for_class_count[class_idx] < max_samples_per_class
                has_available_samples = next_sample_idx_for_class[class_idx] < len(tag_idx_to_img_paths[class_idx])

                if can_add_more_for_this_class and has_available_samples:
                    selected_img_path = tag_idx_to_img_paths[class_idx][next_sample_idx_for_class[class_idx]]
                    txt_path = img_path_to_txt_path_map.get(selected_img_path)
                    if txt_path:
                        self.file_pairs.append((selected_img_path, txt_path))
                        samples_added_for_class_count[class_idx] += 1
                        next_sample_idx_for_class[class_idx] += 1
                        num_added_total += 1
                        made_a_selection_this_round = True
                    
                    if samples_added_for_class_count[class_idx] < max_samples_per_class and \
                       next_sample_idx_for_class[class_idx] < len(tag_idx_to_img_paths[class_idx]):
                        temp_active_classes_next_round.append(class_idx)
                    
                    if target_total_train_size > 0 and num_added_total >= target_total_train_size:
                        break 
                elif has_available_samples: 
                    temp_active_classes_next_round.append(class_idx)
            
            active_class_indices = list(set(temp_active_classes_next_round)) 

            if not made_a_selection_this_round and not active_class_indices : 
                break
            if not made_a_selection_this_round and active_class_indices: 
                all_capped = True
                for c_idx in active_class_indices:
                    if samples_added_for_class_count[c_idx] < max_samples_per_class:
                        all_capped = False
                        break
                if all_capped:
                    if self.rank == 0: logger.info("所有剩余活跃类别均已达到其采样上限。")
                    break
        
        if loop_iteration >= safety_break_max_loops and self.rank == 0:
            logger.warning(f"平衡采样循环达到了最大启发式迭代次数 ({safety_break_max_loops})。当前训练集大小: {len(self.file_pairs)}")

        seeded_random.shuffle(self.file_pairs) # 对最终选出的训练集样本进行一次整体的随机打乱
        if self.rank == 0: logger.info(f"平衡采样完成。最终训练集大小: {len(self.file_pairs)}")


    def _apply_standard_splitting(self, all_image_paths):
        """应用标准的数据划分策略（分层或随机）。所有 rank 执行相同的逻辑。"""
        X_filepaths_for_split = []
        y_labels_list_for_split = [] 
        all_valid_file_pairs_map_standard = {} 

        use_stratified_split_config = getattr(self.config, 'ENABLE_STRATIFIED_SPLIT', True) 
        use_stratified_split = use_stratified_split_config and SKMULTILEARN_AVAILABLE
        
        if self.rank == 0 and use_stratified_split_config and not SKMULTILEARN_AVAILABLE:
            logger.warning("配置中 ENABLE_STRATIFIED_SPLIT=True，但 scikit-multilearn 不可用。将回退到随机划分。")
        
        temp_file_pairs_for_random_split = []
        prog_desc_std = "预处理数据用于标准划分"
        disable_tqdm_std = (self.rank != 0 and self.world_size > 1)

        for img_path in tqdm(all_image_paths, desc=prog_desc_std, disable=disable_tqdm_std):
            base_filename, _ = os.path.splitext(img_path)
            txt_path = base_filename + ".txt"
            if os.path.exists(txt_path):
                all_valid_file_pairs_map_standard[img_path] = txt_path
                if use_stratified_split and self.mode != "test": 
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f_txt:
                            tags_str = f_txt.read().strip()
                        current_img_tags = set(tags_str.split(self.config.TAG_SEPARATOR_IN_TXT))
                        current_img_tags = {tag for tag in current_img_tags if tag}
                        
                        label_vector = torch.zeros(self.config.NUM_CLASSES, dtype=torch.int8)
                        if self.tag_to_idx and self.config.NUM_CLASSES > 0:
                            for tag in current_img_tags:
                                if tag in self.tag_to_idx:
                                    label_vector[self.tag_to_idx[tag]] = 1
                        
                        X_filepaths_for_split.append(img_path)
                        y_labels_list_for_split.append(label_vector.numpy())
                    except Exception as e:
                        if self.rank == 0: logger.warning(f"读取标签文件 {txt_path} 出错 (分层准备): {e}")
                else: 
                    temp_file_pairs_for_random_split.append((img_path, txt_path))
        
        y_labels_np = None 
        if use_stratified_split and self.mode != "test":
            if not X_filepaths_for_split:
                if self.rank == 0: logger.error("分层划分：没有找到任何有效的图像-标签对。")
                use_stratified_split = False 
            else:
                X_filepaths_np = np.array(X_filepaths_for_split)
                y_labels_np = np.array(y_labels_list_for_split)
        
        if self.mode == "train" or self.mode == "val":
            val_split_ratio = getattr(self.config, 'VALIDATION_SPLIT_RATIO', 0.2)
            if not (0 < val_split_ratio < 1):
                if self.rank == 0: logger.warning(f"无效的 VALIDATION_SPLIT_RATIO: {val_split_ratio}，将使用默认值 0.2")
                val_split_ratio = 0.2

            train_img_paths, val_img_paths = [], []

            if use_stratified_split and y_labels_np is not None and y_labels_np.shape[0] > 1 and y_labels_np.shape[1] > 0:
                if self.rank == 0: logger.info(f"使用 IterativeStratification 进行数据划分，验证集比例: {val_split_ratio}")
                if y_labels_np.ndim == 1 and self.config.NUM_CLASSES == 1:
                    y_labels_np = y_labels_np.reshape(-1, 1)
                stratifier = IterativeStratification(n_splits=2, order=1, 
                                                     sample_distribution_per_fold=[val_split_ratio, 1.0 - val_split_ratio])
                try:
                    val_indices_strat, train_indices_strat = next(stratifier.split(X_filepaths_np, y_labels_np))
                    train_img_paths = X_filepaths_np[train_indices_strat].tolist()
                    val_img_paths = X_filepaths_np[val_indices_strat].tolist()
                    if self.rank == 0: logger.info(f"分层划分完成：训练集样本数 {len(train_img_paths)}, 验证集样本数 {len(val_img_paths)}")
                except ValueError as e: 
                    if self.rank == 0: logger.error(f"IterativeStratification 执行失败: {e}。将回退到随机划分。")
                    use_stratified_split = False 
            
            if not use_stratified_split: 
                if not temp_file_pairs_for_random_split: 
                    for img_path_rand in all_image_paths: 
                        txt_path_rand = all_valid_file_pairs_map_standard.get(img_path_rand)
                        if txt_path_rand: temp_file_pairs_for_random_split.append((img_path_rand, txt_path_rand))
                
                if not temp_file_pairs_for_random_split:
                     if self.rank == 0: logger.error("随机划分：没有有效的图像-标签对。")
                     return # 如果没有数据，则 file_pairs 为空

                if self.rank == 0: logger.info(f"回退到随机划分。总有效样本数: {len(temp_file_pairs_for_random_split)}, 验证集比例: {val_split_ratio}")
                seeded_random_split = random.Random(BALANCED_SAMPLING_SEED) # 使用一致的种子
                seeded_random_split.shuffle(temp_file_pairs_for_random_split)
                split_point = int(len(temp_file_pairs_for_random_split) * (1.0 - val_split_ratio))
                train_pairs_random = temp_file_pairs_for_random_split[:split_point]
                val_pairs_random = temp_file_pairs_for_random_split[split_point:]
                train_img_paths = [p[0] for p in train_pairs_random]
                val_img_paths = [p[0] for p in val_pairs_random]

            selected_img_paths = train_img_paths if self.mode == "train" else val_img_paths
            for img_path_sel in selected_img_paths:
                txt_path_sel = all_valid_file_pairs_map_standard.get(img_path_sel)
                if txt_path_sel: 
                    self.file_pairs.append((img_path_sel, txt_path_sel))

        elif self.mode == "test":
            if self.rank == 0: logger.info("测试模式：使用所有找到的有效图像-标签对。")
            for img_path_test in all_image_paths: 
                txt_path_test = all_valid_file_pairs_map_standard.get(img_path_test)
                if txt_path_test:
                    self.file_pairs.append((img_path_test, txt_path_test))
        else:
            raise ValueError(f"不支持的模式: {self.mode}.")

    def _find_image_paths(self):
        image_paths = []
        for ext in self.config.IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(self.image_dir, f"*{ext}")))
        image_paths.sort() 
        return image_paths

    def _build_vocab_from_selected_tags(self):
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

        filter_threshold = getattr(self.config, 'FILTER_TAG_COUNT_THRESHOLD', 0)

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
            if self.rank == 0: logger.info("未进行基于 'count' 列的过滤")
        
        tag_to_idx = {tag: idx for idx, tag in enumerate(final_tags_list)}
        
        if not final_tags_list and self.rank == 0:
            logger.warning("警告: 处理 selected_tags.csv 后，词汇表为空")

        if self.rank == 0: logger.info(f"最终构建的词汇表包含 {len(final_tags_list)} 个标签")
        return final_tags_list, tag_to_idx

    def _get_transform(self):
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
        return len(self.file_pairs)

    def __getitem__(self, idx):
        if idx >= len(self.file_pairs): 
            raise IndexError(f"索引 {idx} 超出数据集范围 {len(self.file_pairs)}")

        img_path, txt_path = self.file_pairs[idx]
        
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
            if self.rank == 0: logger.warning(f"警告: 标签文件 {txt_path} 在 __getitem__ 中未找到")
            current_img_tags = set() 
        except Exception as e:
            if self.rank == 0: logger.error(f"读取或解析标签文件 {txt_path} 时出错: {e}")
            current_img_tags = set()
        
        target = torch.zeros(self.config.NUM_CLASSES, dtype=torch.float32) 
        if not self.tag_to_idx and self.config.NUM_CLASSES > 0 : 
             if self.rank == 0: logger.warning(f"图像 {img_path} 的标签处理跳过，因为 tag_to_idx 为空")
        elif self.tag_to_idx and self.config.NUM_CLASSES > 0: 
            for tag in current_img_tags:
                if tag in self.tag_to_idx: 
                    target[self.tag_to_idx[tag]] = 1.0
        
        return image, target

def get_dataloader(config, mode="train", tags_list=None, tag_to_idx=None, rank=0, world_size=1):
    dataset = DanbooruDataset(config, mode, tags_list, tag_to_idx, rank, world_size)
    
    if len(dataset) == 0:
        if rank == 0:
            logger.warning(f"模式 '{mode}' 的数据集为空，返回一个空的 DataLoader。")
        return DataLoader(dataset, batch_size=max(1, config.BATCH_SIZE), shuffle=False, num_workers=0), dataset.tags_list, dataset.tag_to_idx

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
    
    num_workers_val = max(0, os.cpu_count() // (2 * max(1,world_size)) if os.cpu_count() else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=max(1, config.BATCH_SIZE), 
        shuffle=shuffle_in_loader,
        num_workers=num_workers_val,
        # 在多卡版本中，config.DEVICE 是 rank (int)
        pin_memory=True if isinstance(config.DEVICE, int) else False, 
        drop_last=(mode == "train" and len(dataset) > 0), 
        sampler=sampler
    )
    
    return dataloader, final_tags_list, final_tag_to_idx
