# Image -> Feature Vector

import copy
import torch
import torch.nn as nn
from torchvision import models
from typing import List


def get_output_size(model: nn.Module, image_size=224, channels=3):
    """获取一个编码器模型的输出尺寸"""
    input_tensor = torch.randn(1, channels, image_size, image_size)
    with torch.no_grad():
        output = model(input_tensor)
    return list(output.shape[1:])


class CNN_FC(nn.Module):
    """编码器加上FC层的模型"""

    def __init__(
        self,
        encoder: nn.Module,  # 纯CNN模型
        fc_layers: List,  # 每个全连接层的输出维度，如 [1500,100]
        image_size=224,
        channels=3,
        record_test_info=False,  # 记录测试信息
    ):
        super(CNN_FC, self).__init__()
        self.encoder = copy.deepcopy(encoder)
        input_size = get_output_size(self.encoder, image_size, channels)
        num_features = 1
        for dim in input_size:
            num_features *= dim
        fc = nn.Sequential()
        in_features = num_features
        for i, out_features in enumerate(fc_layers):
            fc.add_module(f"fc_{i}", nn.Linear(in_features, out_features))
            if i < len(fc_layers) - 1:
                fc.add_module(f"relu_{i}", nn.ReLU(inplace=True))
            in_features = out_features
        self.fc = fc
        self.record_test_info = record_test_info
        self.test_info = {
            "fc_input": None,
        }

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        if self.record_test_info:
            self.test_info["fc_input"] = x.clone().detach()
        x = self.fc(x)
        return x

    def get_params(self):
        """获取所有层的当前参数"""
        params = []
        for param in self.parameters():
            params.append(param.clone())
        return params

    def get_grads(self, noise_scale=0.0):
        """获取所有层的当前梯度。noise_scale控制添加的高斯噪声强度。"""
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grad_clone = param.grad.clone()
                if noise_scale > 0:
                    noise = torch.randn_like(grad_clone) * noise_scale
                    grads.append(grad_clone + noise)
                else:
                    grads.append(grad_clone)
            else:
                grads.append(None)
        return grads

    def get_fc_1_w_index(self):
        """获取第1个FC层的权重在 parameters 中的下标"""
        for index, (name, _) in enumerate(self.named_parameters()):
            if "fc_0.weight" in name:
                return index
        return None

    def get_fc_1_w_grad(self):
        """获取第1个FC层的权重梯度"""
        first_fc_layer = self.fc[0]
        if isinstance(first_fc_layer, nn.Linear):
            return (
                first_fc_layer.weight.grad.clone()
                if first_fc_layer.weight.grad is not None
                else None
            )
        return None

    def get_fc_1_input(self):
        """获取第1个FC层在上一轮训练的输入张量"""
        return self.test_info["fc_input"]


def resnet50(pool: bool = False):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if not pool:
        model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()
    return model


def resnet18(pool: bool = False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if not pool:
        model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()
    return model


encoders = {
    "resnet18": resnet18,
    "resnet50": resnet50,
}
