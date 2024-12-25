import torch
import torch.nn as nn
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Importing `peak_signal_noise_ratio` from `torchmetrics.functional` was deprecated",
)

try:
    from torchmetrics.image import peak_signal_noise_ratio
except ImportError:
    from torchmetrics.functional import peak_signal_noise_ratio

from .pytorch_ssim import ssim as ssim_

_func_mse = nn.MSELoss()


def mse(tensor1, tensor2):
    return _func_mse(tensor1, tensor2).item()


def psnr(tensor1, tensor2):
    return peak_signal_noise_ratio(tensor1, tensor2, dim=0, data_range=1.0).item()


def ssim(tensor1, tensor2):
    return ssim_(tensor1, tensor2).item()


def mse_psnr_ssim(tensor1, tensor2):
    return mse(tensor1, tensor2), psnr(tensor1, tensor2), ssim(tensor1, tensor2)


# 计算两个张量A和B的余弦相似度，考虑样本顺序可能不同。
def cosine_similarity(A: torch.Tensor, B: torch.Tensor):
    """
    A (torch.Tensor): 形状为 [m, d] 的张量，表示m个样本，每个样本d个参数。
    B (torch.Tensor): 形状为 [n, d] 的张量，表示n个样本，每个样本d个参数。
    返回:
    avg_similarity (float): 总相似度。
    matches (List[Tuple[int, int, float]]): 每个样本的匹配结果，包含 (A样本索引, B样本索引, 相似度)。
    """
    # 如果不是二维，则转为二维
    if A.ndimension() > 2:
        A = A.view(A.shape[0], -1)
        B = B.view(B.shape[0], -1)
    # 计算余弦相似度矩阵
    cos_sim_matrix = torch.mm(A, B.t()) / (
        torch.norm(A, dim=1).unsqueeze(1) * torch.norm(B, dim=1).unsqueeze(0)
    )
    # 检查并处理 NaN 和 Inf 值
    if torch.isnan(cos_sim_matrix).any():
        cos_sim_matrix = torch.nan_to_num(cos_sim_matrix, nan=0.0)
    if torch.isinf(cos_sim_matrix).any():
        cos_sim_matrix = torch.nan_to_num(cos_sim_matrix, posinf=0.0, neginf=0.0)
    # 将相似度矩阵转换为CPU上的numpy数组
    cos_sim_matrix_np = cos_sim_matrix.cpu().numpy()
    # 使用匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(-cos_sim_matrix_np)
    # 计算平均相似度
    avg = cos_sim_matrix_np[row_ind, col_ind].sum() / len(A)
    # 构建匹配结果
    matches = [(i, j, cos_sim_matrix_np[i, j]) for i, j in zip(row_ind, col_ind)]
    # 整理为按 B 升序
    matches = sorted(matches, key=lambda x: x[1])
    return avg, matches


# 根据 cosine_similarity 匹配结果调整张量A的顺序，使其与张量B一致。
def reorder_tensor(
    A: torch.Tensor,
    matches: List[Tuple[int, int, float]],
    B: torch.Tensor = None,  # 可选，如果传入 B ，且与 A 长度不一致，而删除二者的未匹配的元素。
):
    # 提取匹配的索引
    a_indices, b_indices, _ = zip(*matches)
    # 无需调整形状
    if a_indices == b_indices:
        if B is None:
            print("==", a_indices, b_indices)
            return A
        if A.shape == B.shape:
            return A, B

    # 根据匹配结果调整 A 的顺序
    reordered_A = A[list(a_indices)]
    if B is None:
        return reordered_A
    # 调整 B 的顺序
    reordered_B = B[list(b_indices)]
    return reordered_A, reordered_B


# 将A的顺序调整为与B一致，返回 A B 余弦相似度
def cosine_reorder(A: torch.Tensor, B: torch.Tensor):
    avg_cos, matches = cosine_similarity(A, B)
    rA, rB = reorder_tensor(A, matches, B)
    return rA, rB, avg_cos
