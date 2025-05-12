import torch
import os

class BaseConfig:
    # 路径配置
    # 获取项目根目录
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 数据集目录
    DATA_DIR = os.path.join(PROJECT_ROOT, "data/")
    # 图片和对应的.txt标签文件都存放在IMAGE_DIR下，这个目录在DATA_DIR下
    IMAGE_DIR = os.path.join(DATA_DIR, "images/")
    # 存放所有标签的csv文件，在DATA_DIR下
    SELECTED_TAGS_CSV = os.path.join(DATA_DIR, "selected_tags.csv")
    
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "trained_models/")
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs/")

    # 数据集配置
    IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"] # 支持的图片文件扩展名
    TAG_SEPARATOR_IN_TXT = ',' # .txt 标签文件中标签之间的分隔符
    IMAGE_SIZE = 224 # 输入模型的图像尺寸
    NORM_MEAN = [0.485, 0.456, 0.406] # ImageNet 均值
    NORM_STD = [0.229, 0.224, 0.225] # ImageNet 标准差
    
    # 标签词汇表控制
    # 仅当 selected_tags.csv 中存在 'count' 列时生效
    # 如果标签的 'count' 值小于此阈值，则该标签被过滤掉，不参与训练
    # 设置为 None 或 0 表示不使用此阈值进行过滤
    FILTER_TAG_COUNT_THRESHOLD = 50 

    # 模型配置
    MODEL_NAME = "resnet50" # 使用 timm 库中的模型名称
    PRETRAINED = True # 是否加载 ImageNet 预训练权重 (仅在从头训练或首次微调时有效)
    NUM_CLASSES = -1 # 将由数据集准备阶段根据词汇表动态计算并设置
    DROP_RATE = 0.1 # 模型中的 Dropout rate
    DROP_PATH_RATE = 0.1 # Stochastic depth drop rate (for models like Swin)

    # 训练配置
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 50 # 总训练轮数
    BATCH_SIZE = 32 # 训练批次大小
    ACCUMULATION_STEPS = 1 # 梯度累积步数 ( efektif_batch_size = BATCH_SIZE * ACCUMULATION_STEPS )
    LEARNING_RATE = 1e-4 # 初始学习率
    WEIGHT_DECAY = 1e-5 # 权重衰减
    LR_SCHEDULER = "CosineAnnealingLR" # 学习率调度器: "StepLR", "CosineAnnealingLR", "ReduceLROnPlateau", None
    LR_SCHEDULER_PARAMS = {"T_max": EPOCHS, "eta_min": 1e-6} if LR_SCHEDULER == "CosineAnnealingLR" else \
                          {"step_size": 10, "gamma": 0.1} if LR_SCHEDULER == "StepLR" else \
                          {"factor":0.1, "patience":5} if LR_SCHEDULER == "ReduceLROnPlateau" else {}
    OPTIMIZER = "AdamW" # 优化器: "AdamW", "SGD"
    LOSS_FN = "BCEWithLogitsLoss" # 损失函数: "BCEWithLogitsLoss", "AsymmetricLoss"
    LOSS_FN_PARAMS = {} # 损失函数的参数, 例如 ASL 的 gamma_pos, gamma_neg

    # 高效注意力机制
    # SDPA 可以利用 FlashAttention,xformers 等后端（如果硬件和输入兼容）
    ENABLE_SDP_ATTENTION = True 

    # 检查点配置
    SAVE_BEST_ONLY = True # 是否只保存验证集上性能最好的模型
    SAVE_INTERVAL = 1 # 如果 SAVE_BEST_ONLY=False，每隔多少个 epoch 保存一次模型

    # 推理配置
    INFERENCE_THRESHOLD = 0.35 # 预测时，标签置信度高于此阈值才输出

    def __init__(self):
        # 确保输出目录存在
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.IMAGE_DIR, exist_ok=True) # 确保图片目录存在，方便用户创建

    def to_dict(self):
        """将配置转换为字典，方便保存和加载"""
        return {key: getattr(self, key) for key in dir(self) if not key.startswith('_') and not callable(getattr(self, key))}

    @classmethod
    def from_dict(cls, config_dict):
        """从字典加载配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

