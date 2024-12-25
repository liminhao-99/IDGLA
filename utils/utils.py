import math
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import List


def init_seed(seed=0):
    """初始化随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保PyTorch的计算是可重复的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def standardization(x: torch.Tensor):
    """对张量 [n, m] 中， n 组样本进行单独的标准化，使其均值为0标准差为1。"""
    means = x.mean(dim=1, keepdim=True)
    stds = x.std(dim=1, keepdim=True)
    standardized = (x - means) / stds
    return standardized


def normalization(x: torch.Tensor):
    """对张量 [n, m] 中， n 组样本进行单独的归一化，使其模长为1。"""
    norms = x.norm(dim=1, keepdim=True)  # 计算每组样本的模
    normalized = x / norms  # 将每个样本除以其模
    return normalized


def visualize(
    batch_tensor: torch.Tensor | list,  # 图像张量或图像张量的列表
    label="Sample",  # 展示标签前缀
    max_num=10,  # 展示的数量上限
    save_path="",  # 图像保存位置，空为不保存
    is_show=True,  # 是否展示图像
    is_label=True,  # 是否显示标签
    columns=8,  # 列数
):
    """可视化一组图像张量或特征向量
    - batch_tensor (torch.Tensor): 包含图像数据或特征向量的张量，形状 (batch_size, ...)。
    - label (str): 展示标签前缀。
    - max_num (int): 展示的图像数量上限。
    - save_path (str): 图像保存的路径。如果为空，则不保存图像。
    - is_show (bool): 是否展示图像。如果为 True，则展示图像。
    """
    if isinstance(batch_tensor, list):  # 如果输入是列表，转换为张量
        batch_tensor = torch.stack(batch_tensor)
    batch_tensor = batch_tensor.cpu()
    n = batch_tensor.shape[0]
    image_sizes_dim1 = {
        512: (32, 16),
        1024: (32, 32),
        2048: (64, 32),
        100352: (512, 196),
    }
    image_size = None
    if batch_tensor.dim() == 2:  # 1维张量
        image_size = image_sizes_dim1.get(batch_tensor.shape[1], None)
    elif 3 <= batch_tensor.dim() <= 4:  # 3维张量
        image_size = "img"
        tensor2image = transforms.ToPILImage()
        if batch_tensor.dim() == 3:
            batch_tensor = batch_tensor.unsqueeze(0)
            n = 1
        batch_tensor = batch_tensor.clamp(0, 1)
    assert image_size, f"Unsupported input shape {batch_tensor. shape}"
    if n > max_num:
        n = max_num
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3 * rows))
    for i in range(n):
        ax = axes[i // columns, i % columns] if rows > 1 else axes[i % columns]
        if is_label:
            ax.set_title(f"{label} {i+1}")
        ax.axis("off")  # 关闭坐标轴显示
        if image_size == "img":
            img = tensor2image(batch_tensor[i].squeeze())
            ax.imshow(img, cmap="gray")
        else:
            # 将 1D 张量重整形为 2D 张量
            tensor_reshaped = batch_tensor[i].view(*image_size)
            # 归一化到 0-1 范围
            tensor_normalized = tensor_reshaped - tensor_reshaped.min()
            tensor_normalized /= tensor_reshaped.max() - tensor_reshaped.min()
            tensor_numpy = tensor_normalized.numpy()
            ax.imshow(tensor_numpy, cmap="viridis")
    # 如果样本数不是列数的倍数，隐藏多余的子图
    if n % columns != 0:
        for j in range(n, rows * columns):
            fig.delaxes(
                axes[j // columns, j % columns] if rows > 1 else axes[j % columns]
            )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if is_show:
        plt.show()
    return plt


def cosine_distance(A: List, B: List):
    """两组梯度的余弦相似度"""
    A_flatten = torch.cat([x.view(-1) for x in A]).view(-1)
    B_flatten = torch.cat([y.view(-1) for y in B]).view(-1)
    costs = torch.dot(A_flatten, B_flatten) / (A_flatten.norm() * B_flatten.norm())
    costs = 1 - costs
    return costs


def total_variation(A: torch.Tensor):
    """TV正则项"""
    dx = torch.mean(torch.abs(A[:, :, :, :-1] - A[:, :, :, 1:]))
    dy = torch.mean(torch.abs(A[:, :, :-1, :] - A[:, :, 1:, :]))
    return dx + dy


def compress_gradient(gradient_tensor, compression_rate):
    """
    压缩梯度张量

    参数：
    gradient_tensor (torch.Tensor): 输入的梯度张量
    compression_rate (float): 压缩率，取值范围 (0, 1)，表示保留的比例

    返回：
    torch.Tensor: 压缩后的梯度张量
    """
    if compression_rate == 1:
        return gradient_tensor
    if not (0 < compression_rate < 1):
        raise ValueError("Compression rate must be between 0 and 1.")

    # 计算保留的元素数量
    num_elements = gradient_tensor.numel()
    num_to_keep = int(num_elements * compression_rate)

    # 获取绝对值最大的元素的索引
    _, indices = torch.topk(torch.abs(gradient_tensor.flatten()), num_to_keep)

    # 创建一个全零的张量用于保存压缩后的梯度
    compressed_gradient = torch.zeros_like(gradient_tensor)

    # 将保留的元素放回到压缩后的张量中
    compressed_gradient.flatten()[indices] = gradient_tensor.flatten()[indices]

    return compressed_gradient
