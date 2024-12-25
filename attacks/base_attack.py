import time
import torch

from utils.evaluation import mse_psnr_ssim, cosine_reorder


# 攻击基类
class Base_Attack:
    def __init__(self, device="cpu"):
        self.other_infos = {}
        self.device = device
        self.dummy_x = None  # 缓存 assess 最近一轮的重构样本

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
        self.dummy_x = None  # 重构的样本
        return self.dummy_x

    def assess(
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
        t0 = time.time()
        # 获取伪样本
        dummy_x = self.run(
            grads,
            x,
            y,
            fl_model,
            fc_1_w_grad,
            fc_1_b_grad,
            fc_1_input,
            batch_size,
        )
        # 兼容性检查：去除nan，限定范围0~1
        dummy_x = torch.nan_to_num(dummy_x, nan=0.0)
        dummy_x = dummy_x.clamp(0, 1)
        t1 = time.time()
        # 伪样本重新排序
        dummy_x_re, x_re, _ = cosine_reorder(dummy_x, x)
        self.dummy_x = dummy_x_re
        # 评估指标
        mse, psnr, ssim = mse_psnr_ssim(dummy_x_re, x_re)
        res = {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "time": t1 - t0,
        }
        res.update(self.other_infos)
        return res
