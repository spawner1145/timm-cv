from .base_config import BaseConfig

class Eva02LargeClip336Merged2BConfig(BaseConfig):
    # 数据集配置
    IMAGE_SIZE = 336 # 确认模型使用的输入尺寸
    
    # 使用 CLIP 特定的归一化参数
    # 参考: https://github.com/openai/CLIP/issues/20
    NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
    NORM_STD = [0.26862954, 0.26130258, 0.27577711]

    # 标签词汇表控制
    # 根据你的数据集调整，如果标签数量很多且部分非常稀疏，可以适当提高
    FILTER_TAG_COUNT_THRESHOLD = 20  # 会直接不管总数少于20的标签

    # 模型配置
    MODEL_NAME = "eva02_large_patch14_clip_336.merged2b" # 确切的模型名称，你可以通过运行get_modelnames.py来获取
    
    PRETRAINED = True # 必须为 True，以加载 CLIP 的预训练权重
    DROP_RATE = 0.0   # 微调大型预训练模型时，dropout 通常设为 0 或较小值
    DROP_PATH_RATE = 0.1 # 可以从一个较小的值开始，例如 0.1，或参考模型原始论文

    # 训练配置
    # 大型模型和较大输入图像通常需要更小的批次大小和学习率
    BATCH_SIZE = 4     # 对于336px输入和Large模型，这是一个保守的起始值，根据你的GPU显存调整
                       # 如果遇到显存不足，可以尝试 2 甚至 1
    ACCUMULATION_STEPS = 8 # 示例：等效批次大小为 4*8=32,如果BATCH_SIZE更小，相应增加此值
    
    LEARNING_RATE = 1e-5 # 微调大型 CLIP 模型常用的学习率范围：5e-6 到 2e-5
    WEIGHT_DECAY = 0.05  # AdamW 常用的权重衰减值
    
    EPOCHS = 20 # 微调时通常不需要像从头训练那么多轮次，可以从较少的轮次开始实验
                # 具体轮数取决于你的数据集大小和收敛速度
    
    LR_SCHEDULER = "CosineAnnealingLR"
    LR_SCHEDULER_PARAMS = {"T_max": EPOCHS, "eta_min": 1e-7} # eta_min 可以设置得非常小

    OPTIMIZER = "AdamW"
    
    LOSS_FN = "AsymmetricLoss" # 对于多标签不平衡问题，ASL 是一个很好的选择
    LOSS_FN_PARAMS = {"gamma_pos": 1, "gamma_neg": 4, "clip": 0.05} # ASL 的典型参数

    # 检查点配置
    SAVE_BEST_ONLY = True # 通常微调时更关注最佳性能模型，你如果要过几个epoch保存，设置为 False
    SAVE_INTERVAL = 1 # 如果 SAVE_BEST_ONLY = False，则每隔多少个epoch保存一次模型

    # 其他参数 (如 DEVICE, ENABLE_SDP_ATTENTION, INFERENCE_THRESHOLD 等) 继承自 BaseConfig
    # 你可以根据需要在此处覆盖它们
    # 例如，对于较大的输入和模型，可以考虑将推理阈值稍微调整
    INFERENCE_THRESHOLD = 0.35
