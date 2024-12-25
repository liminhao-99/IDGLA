# Feature Vector -> Image

import os
import torch
import torch.nn as nn
from torchvision import models

from utils.log import logger


class CustomModel(nn.Module):
    # 解码器基类，增加保存和读取参数接口

    def __init__(self, data_root="./data/models", model_name=""):
        super(CustomModel, self).__init__()
        self.data_root = data_root
        self.model_name = model_name
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        if self.model_name:
            self.model_path = os.path.join(self.data_root, self.model_name) + ".pth"
        else:
            self.model_path = None

    def save_model(self):
        if self.model_path:
            try:
                torch.save(self.state_dict(), self.model_path)
                logger.info(f"Model saved to {self.model_path}")
            except Exception:
                logger.error("Failed to save model.", exc_info=True, stack_info=True)
        else:
            logger.info("Model name is empty. Cannot save model.")

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.load_state_dict(torch.load(self.model_path, weights_only=True))
                logger.info(f"Model loaded from {self.model_path}")
            except Exception:
                logger.error("Failed to load model.", exc_info=True, stack_info=True)
        else:
            logger.warning(
                "Model path does not exist or model name is empty. Cannot load model."
            )

    def set_requires_grad(self, requires_grad=True):
        for param in self.parameters():
            param.requires_grad = requires_grad


# Fast Generation-Based Gradient Leakage Attacks
# https://github.com/pigeon-dove/FGLA


class FGLA_resnet(CustomModel):
    class ResConv(nn.Module):
        def __init__(self, channel_size):
            super().__init__()
            self.act = nn.LeakyReLU()
            self.conv = nn.Sequential(
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
                self.act,
                nn.Conv2d(channel_size, channel_size, 3, padding=1, bias=False),
                nn.BatchNorm2d(channel_size),
            )

        def forward(self, x):
            out = self.conv(x) + x
            return self.act(out)


class FGLA_resnet50(FGLA_resnet):

    def __init__(self, *args, **kwargs):
        super(FGLA_resnet50, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            # 14 * 14
            nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(1024),
            self.ResConv(1024),
            self.ResConv(1024),
            # 28 * 28
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(512),
            self.ResConv(512),
            self.ResConv(512),
            # 56 * 56
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),
            # 112 * 112
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),
            # 224 * 224
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 2048, 7, 7)
        out = self.conv(out)
        return out


class FGLA_resnet18(FGLA_resnet):

    def __init__(self, *args, **kwargs):
        super(FGLA_resnet18, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            # 14 * 14
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            self.ResConv(256),
            # 28 * 28
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            self.ResConv(128),
            # 56 * 56
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            self.ResConv(64),
            # 112 * 112
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(32),
            self.ResConv(32),
            self.ResConv(32),
            # 224 * 224
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(16),
            self.ResConv(16),
            self.ResConv(16),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 512, 7, 7)
        out = self.conv(out)
        return out


class FGLA_resnet18_2x(FGLA_resnet):

    def __init__(self, *args, **kwargs):
        super(FGLA_resnet18_2x, self).__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            # 14 * 14
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(256),
            self.ResConv(256),
            # 28 * 28
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(128),
            self.ResConv(128),
            # 56 * 56
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(64),
            self.ResConv(64),
            # 112 * 112
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(32),
            self.ResConv(32),
            # 224 * 224
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            self.ResConv(16),
            self.ResConv(16),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(-1, 512, 7, 7)
        out = self.conv(out)
        return out


decoders = {
    "resnet18": FGLA_resnet18,
    "resnet50": FGLA_resnet50,
}
