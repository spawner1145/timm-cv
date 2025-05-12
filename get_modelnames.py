"""
测试用脚本,查找timm库所有可用的模型名称
"""
import timm
# 尝试查找 EVA-02 Large, patch14, CLIP 预训练, 336px 输入的模型
available_models = timm.list_models('*eva02*large*patch14*clip*336*', pretrained=True)
if not available_models:
    print("未直接找到 eva02_large_patch14_clip_336 的特定 CLIP 预训练模型。尝试更广泛的搜索或检查timm版本。")
    print("可用的 EVA-02 Large CLIP 模型 (可能主要是224px):")
    print(timm.list_models('*eva02*large*patch14*clip*', pretrained=True))
    print("\n可用的 ViT-L/14 CLIP @ 336px (作为替代方案):")
    print(timm.list_models('vit_large_patch14_clip_336.openai', pretrained=True))
else:
    print("找到以下 EVA-02 Large CLIP @ 336px (或相关) 模型:")
    for model_name in available_models:
        print(model_name)