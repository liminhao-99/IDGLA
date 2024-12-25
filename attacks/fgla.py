# Fast Generation-Based Gradient Leakage Attacks: An Approach to Generate Training Data Directly From the Gradient
# https://ieeexplore.ieee.org/abstract/document/10505158
# https://github.com/pigeon-dove/FGLA

import torch

from .base_attack import Base_Attack
from models.decoder import decoders
from utils.evaluation import cosine_reorder, mse


def feature_separation(g_w, g_b, y, offset: bool):
    bz = len(y)
    if offset:
        offset_w = (
            torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(
                dim=0
            )
            * (bz - 1)
            / bz
        )
        offset_b = (
            torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean()
            * (bz - 1)
            / bz
        )
        conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)
    else:
        conv_out = g_w[y] / g_b[y]
    conv_out[torch.isnan(conv_out)] = 0.0
    conv_out[torch.isinf(conv_out)] = 0.0
    return conv_out


def run_fgla(
    generator,  # 预训练解码器
    classification_layer_w_grad,
    classification_layer_b_grad,
    labels,
    offset=True,  # 抵消量
    device="cpu",
):
    conv_out = feature_separation(
        classification_layer_w_grad, classification_layer_b_grad, labels, offset
    )
    conv_out, generator = conv_out.to(device), generator.to(device)
    imgs = generator(conv_out).detach()
    return imgs


class FGLA(Base_Attack):
    def __init__(
        self,
        decoder_class,  # 解码器类名
        decoder_name,  # 解码器参数名称
        offset=True,  # 抵消量
        device="cpu",
    ):
        super().__init__(device)
        self.generator = decoders[decoder_class](model_name=decoder_name)
        self.generator.load_model()
        self.generator.eval()
        self.generator.set_requires_grad(False)
        self.generator = self.generator.to(device)
        self.offset = offset

    def run(
        self,
        grads,  # 梯度，或参数更新
        x,  # 真样本，用于比较
        y,  # 真标签
        fl_model,  # FL全局模型
        fc_1_w_grad,  # 第1个FC层权重梯度
        fc_1_b_grad,  # 第1个FC层偏置梯度
        fc_1_input,  # 第1个FC层输入
        batch_size,  # 批大小
    ):
        # 特征分离
        dummy_z = feature_separation(fc_1_w_grad, fc_1_b_grad, y, self.offset)
        # 特征向量重新排序
        dummy_z_re, z_re, avg_cos = cosine_reorder(dummy_z, fc_1_input)
        # 测量特征向量
        self.other_infos = {
            "feature_cos_sim": avg_cos,
            "feature_mse": mse(dummy_z_re, fc_1_input),
        }
        dummy_x = self.generator(dummy_z_re.to(self.device))
        return dummy_x
