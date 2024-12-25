import math
from copy import deepcopy
from torch import nn, optim

from models import encoder
from utils.datasets import get_dataloader


def data_iterator(
    model: encoder.CNN_FC,  # 模型
    dataloader=None,  # 数据加载器，如果传入了，则忽略以下参数
    dataset_name="cifar100",  # 数据集名称
    batch_size=10,  # 每个客户端的批大小
    batch_num=1,  # 每个客户端的批数量
    local_epochs=2,  # 每个客户端的本地训练轮次
    local_lr=0.1,  # 每个客户端的本地学习率
    image_size=224,  # 图像尺寸
    is_train=False,  # 是否训练集
    shuffle=False,
    step_item=0,
    device="cpu",
):
    if dataloader is None:
        dataloader = get_dataloader(
            dataset_name, batch_size, shuffle, image_size, train=is_train
        )
    dataloader = iter(dataloader)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    ds_len = len(dataloader)
    local_num = math.floor(ds_len / batch_num)  # 客户端数量
    init_params = model.get_params()  # 初始参数
    fc_1_w_index = model.get_fc_1_w_index()  # 线性层1的下标
    for _ in range(step_item):
        next(dataloader)
    # 遍历每个客户端
    for local_i in range(local_num):
        local_model = deepcopy(model)  # 本地模型
        optimizer = optim.SGD(local_model.parameters(), lr=local_lr)  # 本地优化器
        # 缓存客户端的本地数据
        batch_datas = []
        for _ in range(batch_num):
            batch_datas.append(next(dataloader))
        batch_fc_1_inputs = []
        # 执行本地轮次
        for ep in range(local_epochs):
            for images, labels in batch_datas:
                images, labels = images.to(device), labels.to(device)
                # 更新本地模型
                y_pred = local_model(images)
                loss = criterion(y_pred, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_fc_1_inputs.append(local_model.get_fc_1_input())
        local_params = local_model.get_params()  # 更新后的本地参数
        # 计算参数更新
        local_update = [
            (p1 - p0).clone().detach() for p0, p1 in zip(init_params, local_params)
        ]

        yield {
            "batch_images": [img.clone().detach() for img, _ in batch_datas],
            "batch_labels": [lab.clone().detach() for _, lab in batch_datas],
            "params_update": local_update,
            "batch_fc_1_inputs": batch_fc_1_inputs,
        }
