# timm库通用danbooru标签模型训练脚本

**项目结构**

```
danbooru_tagger/
├── configs/                      # 配置文件目录
│   ├── base_config.py            # 基础配置文件
│   └── example_swin_config.py    # Swin Transformer 模型特定配置示例
├── data/                         # 数据存放目录 (用户自行准备)
│   ├── images/                   # 存放图片和对应的.txt标签文件
│   │   ├── image1.jpg
│   │   ├── image1.txt
│   │   ├── image2.png
│   │   ├── image2.txt
│   │   └── ...
│   └── selected_tags.csv         # 筛选后的标签列表，用于构建词汇表
├── src/                          # 源代码目录
│   ├── dataset.py                # 数据集加载和预处理
│   ├── model.py                  # 模型定义与分类头适配逻辑
│   ├── losses.py                 # 损失函数 (例如 AsymmetricLoss)
│   ├── utils.py                  # 工具函数 (日志、检查点、评估指标)
├── train.py                      # 训练、微调、继续训练脚本
├── inference.py                  # 推理脚本
├── trained_models/               # 训练好的模型检查点存放目录
└── requirements.txt              # Python 项目依赖  

```
**前排提示**

目前的代码默认是会用分层划分，对大数据集可能会要求极高的系统内存量，如果撑不住，用随机划分，去`src/dataset.py`找到开头的
```
try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True  # <--- 如果你把这里改成 False，就可以强制使用随机划分，大幅减少内存消耗
except ImportError:
    SKMULTILEARN_AVAILABLE = False
```

**一、 环境设置**

根目录运行：

```
pip install -r requirements.txt
```

**二、 数据准备(你可以直接运行data_prepare.py来从danbooru2024数据集获取)**

> 注意:如果你希望使用data_prepare.py，先去[danbooru2024数据集](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel)下载[metadata.parquet](https://huggingface.co/datasets/deepghs/danbooru2024-webp-4Mpixel/blob/main/metadata.parquet)并把它放到根目录

1. **图像和标签文件:(如果你用data_prepare.py，数据准备这些可以跳过了)**
   * 在项目根目录下创建 `data/images/` 文件夹
   * 将你所有的训练图片放入 `data/images/` 文件夹中
   * 对于每一张图片 (例如 `image1.jpg`)，在**相同目录下**创建一个同名的 `.txt` 文件 (例如 `image1.txt`)
   * 在 `.txt` 文件中，写入该图片对应的 Danbooru 标签，标签之间用`,`分隔 (这个分隔符可以在 `base_config.py` 中的 `TAG_SEPARATOR_IN_TXT` 修改)
     * 示例 `image1.txt` 内容: `1girl,solo,long_hair,red_eyes,school_uniform`
2. **`selected_tags.csv` 文件:**
   * 在 `data/` 目录下创建一个名为 `selected_tags.csv` 的 CSV 文件
   * 这个文件用来定义你的模型最终会学习和预测哪些标签
   * **必须包含 `name` 列** : 这一列包含你希望模型使用的所有标签的名称
   * **可选包含 `count` 列** : 如果这一列存在，并且你在配置文件中设置了 `FILTER_TAG_COUNT_THRESHOLD` (例如设置为 50)，那么在构建词汇表时，只有 `count` 值大于等于 50 的标签才会被采纳，如果 `count` 列不存在，或者阈值设为 `None` 或 0，则 `name` 列中的所有标签都会被考虑
   * 示例 `selected_tags.csv` (包含 count 列):

     ```
     name,count
     1girl,150000
     solo,120000
     long_hair,95000
     short_hair,80000
     red_eyes,45       # 如果 FILTER_TAG_COUNT_THRESHOLD = 50, 这个标签会被过滤掉
     blue_eyes,60000
     original,2000
     touhou,100
     ```
   * 示例 `selected_tags.csv` (不包含 count 列):

     ```
     name
     1girl
     solo
     long_hair
     short_hair
     red_eyes
     blue_eyes
     original
     touhou
     ```
   * **重要** : 只有在 `selected_tags.csv` 中出现（并且通过了可选的 count 过滤）的标签，才会被模型学习，即使你的 `.txt` 文件中包含了其他标签，它们也会被忽略

**三、 配置文件说明 (`configs/`)**

* **`base_config.py`** : 这是基础配置文件，它定义了所有通用的参数，如路径、图像大小、默认模型、学习率等，你可以直接修改它，或者创建新的配置文件继承并覆盖它
* **`example_swin_config.py`** : 这是一个特定于 Swin Transformer 模型的配置示例，它继承自 `BaseConfig` 并修改了一些参数以适应 Swin 模型 (例如模型名称、批次大小、学习率、损失函数等)
* **创建你自己的配置** :

1. 在 `configs/` 目录下复制一份 `example_swin_config.py` (或 `base_config.py`) 并重命名，例如 `my_experiment_config.py`
2. 修改文件中的类名，例如 `MyExperimentConfig(BaseConfig):`
3. 根据你的需求调整参数，例如 `MODEL_NAME` (从 [timm 库](https://huggingface.co/docs/timm/index) 选择), `IMAGE_SIZE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `LOSS_FN` (可以选择 `BCEWithLogitsLoss` 或 `AsymmetricLoss`) 等
4. 特别注意 `FILTER_TAG_COUNT_THRESHOLD` 参数，它配合 `selected_tags.csv` 中的 `count` 列使用

**四、 如何训练模型**

训练脚本是 `src/train.py`，你需要通过命令行参数指定使用的配置文件和运行名称

1. **从头开始训练 (或使用 ImageNet 预训练权重):**
   * 确保你的配置文件中的 `PRETRAINED = True` (如果你想从 ImageNet 加载权重) 或 `False` (完全从随机初始化开始)
   * 运行命令:

     ```
     python train.py \
         --config_name eva02_L_clip336_merged2b_config.py \
         --config_class_name Eva02LargeClip336Merged2BConfig \
         --run_name eva_test
     ```

     * `--config_name`: 你在 `configs/` 目录下的配置文件名
     * `--config_class_name`: 配置文件中定义的配置类的名称
     * `--run_name`: 本次运行的唯一标识，日志和模型检查点会以此命名
2. **从检查点继续训练 (Resume Training):**
   * 如果你之前的训练意外中断，或者你想在之前的基础上继续训练相同的任务
   * 这会加载模型的权重、优化器状态、学习率调度器状态以及上次的 epoch 数和词汇表
   * 运行命令:

     ```
     python train.py \
         --config_name eva02_L_clip336_merged2b_config.py \
         --config_class_name Eva02LargeClip336Merged2BConfig \
         --run_name eva_test_resumed \
         --resume_checkpoint trained_models/danbooru_experiment_last.pth.tar
     ```

     * `--resume_checkpoint`: 指向你之前保存的检查点文件路径 (通常是 `*_last.pth.tar` 文件)
     * `--run_name` 可以设置一个新的名称，或者与之前的名称相同（日志会追加）
     * 配置文件 (`--config_name`, `--config_class_name`) 此时主要用于提供一个初始结构，但检查点中的配置和词汇表会优先被加载并覆盖当前配置，以确保一致性
3. **从检查点进行微调 (Finetune):**
   * **场景1: 标签集不变，只是想调整学习率等参数进一步训练**

     * 这与继续训练类似，但优化器和学习率调度器会重新初始化
   * **场景2: 标签集发生变化 (例如，你想让模型学习一些新的标签，或者移除一些旧标签)**

     * 你需要更新你的 `data/selected_tags.csv` 文件以反映新的标签集
     * 模型会加载旧检查点的 backbone 权重
     * 分类头 (classifier head) 会被重新构建以适应新的标签数量
     * 代码会尝试将旧标签对应的分类器权重迁移到新的分类头中，新标签的权重则会重新初始化
     * 优化器和学习率调度器会根据当前配置文件中的设置重新初始化
   * 运行命令:

     ```
     python train.py \
         --config_name eva02_L_clip336_merged2b_config.py \
         --config_class_name Eva02LargeClip336Merged2BConfig \
         --run_name eva_test_finetuned_with_new_tags \
         --finetune_checkpoint trained_models/danbooru_experiment_last.pth.tar
     ```

     * `--finetune_checkpoint`: 指向你希望作为起点的模型检查点
     * `--config_name` 和 `--config_class_name` 应指向你为本次微调任务准备的配置文件 (可能包含新的学习率、epoch 数，并且 `SELECTED_TAGS_CSV` 指向更新后的标签文件)
     * **重要** : `current_run_config` (由 `--config_name` 加载) 定义了微调任务的目标（包括新的词汇表），而 `old_checkpoint_config_dict` (从 `--finetune_checkpoint` 加载) 描述了旧模型的结构，两者共同用于模型适配

**训练过程中的输出:**

* 日志会输出到控制台，并保存在 `trained_models/logs/运行名称.log` 文件中
* 模型检查点会保存在 `trained_models/` 目录下：
  * `运行名称_last.pth.tar`: 每个 epoch 结束时都会保存/覆盖，用于中断后恢复
  * `运行名称_best.pth.tar`: 如果 `SAVE_BEST_ONLY = True` (或即使为 False 但当前是最佳)，则保存验证集上指标最好的模型
  * 如果 `SAVE_BEST_ONLY = False`，还会保存类似 `运行名称_epoch_X.pth.tar` 的文件

**五、 如何进行推理**

使用 `src/inference.py` 脚本对新的图像进行标签预测

* 运行命令:

  ```
  python inference.py \
      --checkpoint_path trained_models/danbooru_experiment_last.pth.tar \
      --input_path /path/to/your/image.jpg 
      # 或者 --input_path /path/to/your/image_folder/
      # --threshold 0.35(可选)
  ```

  * `--checkpoint_path`: 指向你训练好的模型检查点文件 (通常是 `*_best.pth.tar` 或 `*_last.pth.tar`)
  * `--input_path`: 可以是单张图片的路径，也可以是一个包含多张图片的文件夹路径
* **可选参数:**

  * `--device cuda` 或 `--device cpu`: 覆盖检查点配置中的设备设置
  * `--threshold 0.3`: 设置一个新的置信度阈值，覆盖检查点配置中的 `INFERENCE_THRESHOLD`，只有高于此阈值的标签才会被输出
* **输出格式:**
  对于每张图片，输出会是这样的格式：

  ```
  图片: image_name.jpg
  预测标签 (置信度 > 0.5):  <-- 这里的0.5是实际使用的阈值
  tag_name1: 0.9876
  another_tag: 0.7532
  some_other_tag: 0.6543
  ```

---

```
  如果某张图片没有标签的置信度高于阈值，会提示 "没有标签的置信度高于阈值 X.X"
```
