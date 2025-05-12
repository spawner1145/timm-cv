import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss For Multi-Label Classification (ASL)
    参考: https://github.com/Alibaba-MIIL/ASL
    """
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """
        计算非对称损失
        Args:
            x: 模型的原始输出 logits (未经过 sigmoid)，shape: (N, C)
            y: 真实标签 (多标签二元向量)，shape: (N, C)
        Returns:
            loss: 计算得到的损失值
        """
        # 计算概率
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 非对称裁剪 (Asymmetric Clipping)
        if self.clip is not None and self.clip > 0:
            # 对于负样本，稍微提高其概率下限，避免梯度消失过快
            # P_ = P_neg + clip; if P_ > 1 then P_ = 1
            xs_neg = (xs_neg + self.clip).clamp(max=1) 

        # 基础的二元交叉熵计算
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg # 此时 loss 是负值

        # 非对称聚焦 (Asymmetric Focusing)
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                # 暂时禁用梯度计算，以模仿原始 Focal Loss 的行为 (可选)
                torch.set_grad_enabled(False)
            
            # pt 是模型对于正样本的预测概率，或者对于负样本的 (1 - 预测概率)
            # pt = p if y=1 else 1-p
            pt0 = xs_pos * y 
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1

            # 根据正负样本使用不同的 gamma
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            # 聚焦权重 (1-pt)^gamma
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            
            loss *= one_sided_w # 将聚焦权重应用到损失上

        # 返回每个样本的平均损失 (取负号因为标准交叉熵是最小化负对数似然)
        return -loss.sum() / x.size(0) 
        # 或者返回所有元素的平均损失: return -loss.mean()
        # 通常论文中是 sum over classes, mean over batch
        # -loss.sum() / x.size(0) 是每个样本的损失总和的平均值
        # -loss.mean() 是所有元素损失的平均值，两者在多标签中可能略有不同，但影响不大


# 你也可以在这里添加其他自定义损失函数
# 例如，带权重的 BCE Loss
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight # tensor of shape (num_classes,)

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

