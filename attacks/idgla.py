# Improved Generation-Based Gradient Leakage Attacks (iGLA)

import time
import copy
import torch

from .base_attack import Base_Attack
from models.decoder import decoders
from utils.log import logger
from utils.utils import cosine_distance, total_variation, standardization
from utils.ica import ICA
from utils.evaluation import cosine_similarity, mse, cosine_reorder


class IDGLA(Base_Attack):
    def __init__(
        self,
        decoder_class,  # 解码器类名
        decoder_name,  # 解码器参数名称
        ica_num_epochs=4000,  # ICA迭代次数
        is_log=True,  # 输出日志
        device="cpu",
    ):
        super().__init__(device)
        self.decoder = decoders[decoder_class](model_name=decoder_name)
        self.decoder.load_model()
        self.decoder.eval()
        self.decoder.set_requires_grad(False)
        self.decoder = self.decoder.to(device)
        self.is_log = is_log
        self.ica = ICA(num_epochs=ica_num_epochs, is_log=is_log, device=device)
        self.recover_fc_1_input = None

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
        # ICA
        recover_fc_1_input = self.ica.run(fc_1_w_grad, batch_size, fc_1_input)
        # 测试：FedAVG模式下如果 recover_fc_1_input 与 recover_fc_1_input 形状不一致，则取相似度最高的样本
        recover_fc_1_input, fc_1_input, avg_cos = cosine_reorder(
            recover_fc_1_input, fc_1_input
        )
        self.recover_fc_1_input = recover_fc_1_input
        std_re_input = standardization(recover_fc_1_input)  # 标准化
        # 评估
        self.other_infos = {
            "feature_cos_sim": avg_cos,
            "feature_mse": mse(std_re_input, standardization(fc_1_input)),
        }
        # 生成
        dummy_x = self.decoder(std_re_input)
        return dummy_x
