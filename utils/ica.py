"""
Independent Component Analysis , ICA

X = AS

- X 混合信号矩阵（每一列是一个观测向量）。
- A 混合矩阵。
- S 源信号矩阵（每一列是一个独立的源信号）。

S = WX

- W 解混矩阵
"""

import time
import warnings
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD

from utils.log import logger
from utils.evaluation import cosine_similarity, reorder_tensor

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Casting complex values to real discards the imaginary part",
)


class ICA:
    def __init__(
        self,
        lr=0.001,  # 学习率
        num_epochs=4000,  # 优化次数
        # 最优超参
        ne=0.0935827857091549,  # 负熵权重
        decor=0.2513674104161625,  # 去相关权重
        T=16.626277167672217,  # 去相关损失的余弦相似度温度
        nv=0.05811961446881231,  # 负值惩罚权重
        l1=17.94138146247377,  # L1正则权重
        is_log=True,  # 是否输出日志
        device="cpu",
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.ne = ne
        self.decor = decor
        self.T = T
        self.nv = nv
        self.l1 = l1
        self.device = device
        self.eps = torch.tensor(1e-20, device=self.device)
        self.psnr = 0.0
        self.avg_similarity = 0.0
        self.is_log = is_log

    # 负熵损失，最大化非高斯性
    def get_loss_ne(self, estimated_components: torch.Tensor):
        # 计算独立成分的双曲余弦值
        cosh_values = torch.cosh(estimated_components)
        # 取对数
        log_cosh_values = torch.log(cosh_values + self.eps)
        # 求最后一个维度的平均值的平方
        negentropy = log_cosh_values.mean(dim=-1) ** 2
        loss_ne = -negentropy.mean()
        return loss_ne

    # 去相关损失
    def get_loss_decor(self, unmixing_mat_norm: torch.Tensor, T: float):
        # 得到一个对称矩阵，其中每个元素表示解混矩阵的两行之间的余弦相似度。
        cos_matrix = torch.matmul(unmixing_mat_norm, unmixing_mat_norm.T).abs()
        # 计算去相关损失
        loss_decor = (torch.exp(cos_matrix * T) - 1).mean()
        return loss_decor

    # 负值惩罚：解混矩阵和混合信号（梯度）中都含负值，互相抵消，促进源信号中不含负值。
    def get_loss_nv(self, independent_components: torch.Tensor):
        loss_nv = torch.minimum(  # 每个信号中，取正值或负值中较小一方的均值
            F.relu(-independent_components).norm(dim=-1),
            F.relu(independent_components).norm(dim=-1),
        ).mean()
        return loss_nv

    # L1正则：鼓励稀疏性
    def get_loss_l1(self, independent_components: torch.Tensor):
        loss_l1 = torch.abs(independent_components).mean()
        return loss_l1

    # 获取结果
    def get_rec(
        self,
        unmixing_mat: torch.Tensor,  # 解混矩阵
        S_size: int,
        X_w: torch.Tensor,  # 混合信号白化
        X_mean: torch.Tensor,  # 混合信号均值
        V: torch.Tensor,  # 白化矩阵
    ):
        with torch.no_grad():
            # 归一化解混矩阵
            U_norm = unmixing_mat / (unmixing_mat.norm(dim=-1, keepdim=True) + self.eps)
            # 初步计算解混信号
            rec = torch.matmul(U_norm, X_w)
            # 逆中心化处理：调整信号的均值，使其与源信号的均值一致
            rec = rec + torch.matmul(torch.matmul(U_norm, V), X_mean)
            # 恢复输入形状
            rec = rec.detach().view([S_size, -1])
        return rec

    def __call__(
        self,
        X: torch.Tensor,  # 混合信号矩阵
        S_size: int,  # 源信号个数
        S=None,  # 源信号（测试用）
    ):
        return self.run(X, S_size, S)

    def run(
        self,
        X: torch.Tensor,  # 混合信号矩阵
        S_size: int,  # 源信号个数
        S=None,  # 源信号（测试用）
    ):
        X_mean = X.mean(dim=1, keepdims=True)  # 均值
        X_centered = X - X_mean  # 中心化

        # 计算协方差矩阵 C
        C = torch.matmul(X_centered, X_centered.T) / (X_centered.shape[1] - 1)
        # 对 C 特征值分解
        eig_vals, eig_vecs = torch.linalg.eig(C)
        # 提取前 S_size 个最大的特征值，获取特征向量矩阵 U 和对应的特征值 lamb
        topk_indices = torch.topk(eig_vals.float().abs(), S_size)[1]
        U = eig_vecs.float()
        lamb = eig_vals.float()[topk_indices].abs()
        # 特征值的逆平方根，并构造对角矩阵 lamb_inv_sqrt
        lamb_inv_sqrt = torch.diag(1 / (torch.sqrt(lamb) + self.eps)).float()
        # 计算白化矩阵 V
        V = torch.matmul(lamb_inv_sqrt, U.T[topk_indices]).float()
        # 对输入数据进行白化处理
        X_w = torch.matmul(V, X_centered)

        # 解混矩阵，待优化。
        unmixing_mat = torch.eye(S_size, requires_grad=True, device=self.device)
        param_list = [unmixing_mat]
        opt = SGD(param_list, lr=self.lr, weight_decay=0, momentum=0)
        t0 = time.time()
        log_eps = self.num_epochs // 10
        if self.is_log:
            logger.info(
                f"Start ICA. ne: {self.ne:<4}, decor: {self.decor:<4}, \
T: {self.T:<4}, nv: {self.nv:<4}, l1: {self.l1:<4}"
            )
        for i in range(self.num_epochs):
            loss_ne = loss_decor = loss_nv = loss_l1 = torch.tensor(
                0.0, device=self.device
            )
            # 解混矩阵归一化
            unmixing_mat_norm = unmixing_mat / (
                unmixing_mat.norm(dim=-1, keepdim=True) + self.eps
            )
            # 估计的独立成分（解混信号）
            estimated_components = torch.matmul(unmixing_mat_norm, X_w)
            # 还原的独立成分：经过逆中心化处理，使其与源信号的均值一致
            independent_components = estimated_components + torch.matmul(
                torch.matmul(unmixing_mat_norm, V), X_mean
            )
            # =============================
            # 负熵损失
            if self.ne > 0:
                loss_ne = self.get_loss_ne(estimated_components)
            # 去相关损失
            if self.decor > 0:
                loss_decor = self.get_loss_decor(unmixing_mat_norm, self.T)
            # 负值惩罚
            if self.nv > 0:
                loss_nv = self.get_loss_nv(independent_components)
            # L1正则
            if self.l1 > 0:
                loss_l1 = self.get_loss_l1(independent_components)
            # 总损失
            loss = (
                loss_ne * self.ne
                + loss_decor * self.decor
                + loss_nv * self.nv
                + loss_l1 * self.l1
            )
            loss.backward()
            opt.step()
            recover_S = None
            if self.is_log and (
                i == 0 or i % log_eps == log_eps - 1 or i == self.num_epochs - 1
            ):
                if S is None:
                    logger.info(f"{i:<4}, loss: {loss:.4f}, time: {t1-t0:.4f}s")
                else:
                    # 恢复源信号
                    recover_S = self.get_rec(unmixing_mat, S_size, X_w, X_mean, V)
                    recover_S = recover_S.abs()  # 取绝对值
                    self.avg_similarity, matches = cosine_similarity(recover_S, S)
                    t1 = time.time()
                    logger.info(
                        f"ICA:{i+1:>5}, loss: {loss:.4f}, cos_similarity: {self.avg_similarity:.4f}, time: {t1-t0:.4f}s"
                    )
        # 返回恢复的源信号
        recover_S = self.get_rec(unmixing_mat, S_size, X_w, X_mean, V)
        recover_S = recover_S.abs()
        if S is not None:  # 调整顺序
            self.avg_similarity, matches = cosine_similarity(recover_S, S)
            recover_S, _ = reorder_tensor(recover_S, matches, S)  # 重新排列顺序
        return recover_S
