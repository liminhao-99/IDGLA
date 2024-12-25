import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import random

from utils.log import logger

"""
imagenet
caltech256
cifar10
cifar100
"""

DATASETS_DIR = "./data/datasets"


def _read_datasets_dir():
    global DATASETS_DIR
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir_path = os.path.join(current_dir, "datasets_dir.txt")
    try:
        with open(datasets_dir_path, "r", encoding="utf-8") as file:
            DATASETS_DIR = file.readline().strip()
    except Exception:
        pass
    logger.debug(f"DATASETS_DIR: {DATASETS_DIR}")


_read_datasets_dir()


def get_dataset(name: str, image_size: int, train: bool):
    transform = transforms.Compose(
        [transforms.Resize([image_size, image_size]), transforms.ToTensor()]
    )
    if name == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet(
            f"{DATASETS_DIR}/imagenet/", transform=transform, split=split
        )
    elif name == "caltech256":
        dataset = datasets.Caltech256(
            f"{DATASETS_DIR}/caltech256",
            transform=transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.Lambda(
                        lambda x: x.convert("RGB") if x.mode != "RGB" else x
                    ),
                    transforms.ToTensor(),
                ]
            ),
            download=True,
        )
        # 排除不存在文件的index
        dataset.index = (
            dataset.index[:6307] + dataset.index[6308:22619] + dataset.index[22620:]
        )
        dataset.y = dataset.y[:6307] + dataset.y[6308:22619] + dataset.y[22620:]
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            f"{DATASETS_DIR}/cifar10", transform=transform, download=True, train=train
        )
    elif name == "cifar100":
        dataset = datasets.CIFAR100(
            f"{DATASETS_DIR}/cifar100", transform=transform, download=True, train=train
        )
    elif name == "mnist":
        dataset = datasets.MNIST(
            f"{DATASETS_DIR}/mnist",
            transform=transforms.Compose(
                [
                    transforms.Resize([image_size, image_size]),
                    transforms.Lambda(lambda x: x.convert("RGB")),  # 转换为三通道
                    transforms.ToTensor(),
                ]
            ),
            download=True,
            train=train,
        )
    else:
        raise ValueError(f"Dataset '{name}' is not supported.")
    return dataset


def get_dataset_info(name: str):
    info = {
        "imagenet": {
            "num_classes": 1000,
        },
        "caltech256": {
            "num_classes": 257,  # 实际类别
        },
        "cifar10": {
            "num_classes": 10,
        },
        "cifar100": {
            "num_classes": 100,
        },
        "mnist": {
            "num_classes": 10,
        },
    }
    return info.get(name, None)


def get_dataloader(
    name: str, batch_size: int, shuffle=False, image_size=224, train=False
):
    dataset = get_dataset(name, image_size, train)
    dataLoader = DataLoader(dataset, batch_size, shuffle=shuffle, drop_last=True)
    return dataLoader


# 获取指定标签重复率的数据加载器
def get_repetition_dataloader(
    name: str,
    batch_size: int,
    shuffle=False,  # 如果为 True ，则主元素随机
    image_size=224,
    train=False,
    rep_rate=0.1,  # 标签重复率
):
    # 标签重复个数
    rep_num = round(batch_size * rep_rate)
    rep_num = max(0, min(rep_num, batch_size))
    no_rep_num = batch_size - rep_num

    # 取原数据集
    dataset = get_dataset(name, image_size, train)
    dataset_info = get_dataset_info(name)
    num_classes = dataset_info["num_classes"]  # 标签种类数

    # 将原数据集中的样本，打乱顺序后装进对应标签的标签桶
    buckets = [[] for _ in range(num_classes)]
    if shuffle:
        indices = torch.randperm(len(dataset))
    else:
        indices = torch.arange(len(dataset))
    for i in indices:
        d = dataset[i]
        buckets[d[1]].append(i)

    new_order = []  # 新的数据集的下标顺序

    while True:
        if rep_num > 0:
            # 1. 从标签桶中取一个剩余样本数 >= rep_num 的标签，作为主标签
            main_labels = [
                i for i, bucket in enumerate(buckets) if len(bucket) >= rep_num
            ]
            if not main_labels:
                break
            if shuffle:
                main_label = random.choice(main_labels)
            else:
                main_label = main_labels[0]

            # 2. 将 rep_num 个主标签的样本的下标装进 new_order
            main_samples = buckets[main_label][:rep_num]
            new_order.extend(main_samples)
            buckets[main_label] = buckets[main_label][rep_num:]
        else:
            main_label = num_classes + 2

        # 3. 从标签桶中取 no_rep_num 个剩余样本数 >0 的标签，作为次标签
        secondary_labels = [i for i, bucket in enumerate(buckets) if len(bucket) > 0]
        if main_label in secondary_labels:
            secondary_labels.remove(main_label)
        if len(secondary_labels) < no_rep_num:
            break
        if shuffle:
            chosen_secondary_labels = random.sample(secondary_labels, no_rep_num)
        else:
            chosen_secondary_labels = secondary_labels[:no_rep_num]

        # 4. 每个次标签，各取1个样本的下标装进 new_order
        for label in chosen_secondary_labels:
            new_order.append(buckets[label].pop(0))

    # 将 new_order 转换为 new_dataset
    new_dataset = Subset(dataset, new_order)

    dataLoader = DataLoader(new_dataset, batch_size, shuffle=False, drop_last=True)
    return dataLoader


# 获取过滤了标签的数据加载器
def get_filtered_dataloader(
    name: str,
    batch_size: int,
    shuffle=False,
    image_size=224,
    train=False,
    blacklist=[],  # 标签黑名单
    whitelist=[],  # 标签白名单
):
    if not blacklist and not whitelist:
        return get_dataloader(name, batch_size, shuffle, image_size, train)

    dataset = get_dataset(name, image_size, train)
    # 过滤数据集
    filtered_indices = []
    for i in range(len(dataset)):
        label = dataset[i][1]  # 假设 dataset[i] 返回 (image, label) 的元组
        if (whitelist and label not in whitelist) or (label in blacklist):
            continue
        filtered_indices.append(i)

    # 创建过滤后的数据集
    new_dataset = Subset(dataset, filtered_indices)

    # 创建数据加载器
    dataLoader = DataLoader(new_dataset, batch_size, shuffle=shuffle, drop_last=True)
    return dataLoader
