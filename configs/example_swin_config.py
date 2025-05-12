from .base_config import BaseConfig

class ExampleSwinConfig(BaseConfig):
    # 数据集配置
    IMAGE_SIZE = 384 # Swin Transformer 常用的尺寸

    # 标签词汇表控制
    FILTER_TAG_COUNT_THRESHOLD = 10 # 过滤掉出现次数少的标签

    # 模型配置
    MODEL_NAME = "swin_large_patch4_window12_384" # timm 库中的 Swin 模型
    PRETRAINED = True
    DROP_RATE = 0.0 # Swin Transformer 通常在预训练时已包含正则化
    DROP_PATH_RATE = 0.2 # Swin-Base 推荐的 drop path rate

    # 训练配置
    BATCH_SIZE = 16 # Swin Transformer 通常比 ResNet 需要更多显存，可能需要减小批次大小
    ACCUMULATION_STEPS = 2 # 示例：等效批次大小为 16*2=32
    LEARNING_RATE = 5e-5 # Swin Transformer 微调时常用的学习率
    WEIGHT_DECAY = 0.05
    LOSS_FN = "AsymmetricLoss" # 使用 ASL 处理多标签不平衡问题
    LOSS_FN_PARAMS = {"gamma_pos": 1, "gamma_neg": 4, "clip": 0.05} # ASL 的典型参数

    # 其他参数可以继承自 BaseConfig 或在此处覆盖
