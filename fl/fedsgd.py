from torch import nn

from models import encoder
from utils.datasets import get_dataloader


def data_iterator(
    model: encoder.CNN_FC,  # 模型
    dataloader=None,  # 数据加载器，如果传入了，则忽略以下参数
    dataset_name="cifar100",  # 数据集名称
    batch_size=10,  # 批大小
    image_size=224,  # 图像尺寸
    is_train=False,  # 是否训练集
    shuffle=False,  # 是否乱序
    noise_scale=0.0,  # 模拟差分隐私，添加噪声强度
    step_item=0,
    device="cpu",
):
    if dataloader is None:
        dataloader = get_dataloader(
            dataset_name, batch_size, shuffle, image_size, train=is_train
        )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for index, (images, labels) in enumerate(dataloader):
        if index < step_item:
            continue
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        loss.backward()

        yield {
            "images": images.clone().detach(),
            "labels": labels.clone().detach(),
            "grads": model.get_grads(noise_scale=noise_scale),
            "fc_1_input": model.get_fc_1_input(),
        }
