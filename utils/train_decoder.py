import time
from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.image import PeakSignalNoiseRatio

from utils.log import logger
from utils.history import History
from utils.datasets import get_dataloader
from utils.evaluation import mse, psnr, ssim
from utils.utils import normalization, standardization


# 测试解码器
def test(
    encoder: nn.Module,
    decoder: nn.Module,
    testloader: DataLoader,
    pretreatment,
    device="cuda",
):
    decoder.eval()
    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    t0 = time.time()
    with torch.no_grad():
        for inputs, _ in testloader:
            inputs = inputs.to(device)
            encoded = encoder(inputs)
            if pretreatment is not None:
                encoded = pretreatment(encoded)
            outputs = decoder(encoded)
            total_mse += mse(outputs, inputs)
            total_ssim += ssim(outputs, inputs)
            total_psnr += psnr(outputs, inputs)
    len_testloader = len(testloader)
    mse_value = total_mse / len_testloader
    ssim_value = total_ssim / len_testloader
    psnr_value = total_psnr / len_testloader
    t1 = time.time()
    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "time": t1 - t0,
        "last_inputs": inputs,
        "last_outputs": outputs,
    }


# 训练解码器
def train(
    encoder: nn.Module,
    decoder: nn.Module,
    image_size: int,
    train_dataset: str,
    test_dataset: Union[DataLoader, None],
    batch_size: int,
    num_epochs: int,
    lr: float,
    device="cuda",
    use_multi_gpu=False,  # 多GPU
    device_ids=None,  # GPU id，None使用全部，可指定 [0, 1]
    pretreatment=standardization,  # 编码器特征向量的预处理步骤，可选 None standardization normalization
):
    trainloader = get_dataloader(train_dataset, batch_size, image_size, train=True)
    testloader = get_dataloader(test_dataset, batch_size, image_size, train=False)
    train_info = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "batch_size": batch_size,
        "image_size": image_size,
        "lr": lr,
        "encoder": type(encoder).__name__,
        "decoder": type(decoder).__name__,
    }

    encoder = encoder.to(device)
    for param in encoder.parameters():  # 屏蔽编码器梯度更新
        param.requires_grad = False
    decoder = decoder.to(device)
    encoder.train()  # 训练模式，开启BN
    decoder.train()
    model_path = None
    if hasattr(decoder, "model_path") and decoder.model_path:
        # 提取模型位置，存放历史记录
        model_path = decoder.model_path
    if use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        encoder = nn.DataParallel(encoder, device_ids)
        decoder = nn.DataParallel(decoder, device_ids)
    if model_path:
        history_path = model_path + ".train_history.csv"
        history = History(save_path=history_path)
    else:
        history = History()
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logger.info("Start train. Press Ctrl+C to save and exit.")
    t0 = time.time()

    try:
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, _ in trainloader:
                inputs = inputs.to(device)
                with torch.no_grad():
                    encoded = encoder(inputs)
                    if pretreatment is not None:
                        encoded = pretreatment(encoded)
                outputs = decoder(encoded)
                loss = criterion(outputs, inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            t1 = time.time()
            t = t1 - t0
            loss = running_loss / len(trainloader)
            text = f"Epoch [{epoch + 1}/{num_epochs}], time={t:.4f}s, Loss: {loss:.4f}"
            his = {
                "epoch": epoch,
                "loss": loss,
                "time": t,
            }
            if isinstance(decoder, nn.DataParallel):
                decoder.module.save_model()
            else:
                decoder.save_model()
            if testloader:
                res = test(encoder, decoder, testloader, pretreatment, device)
                his["mse"] = res["mse"]
                his["ssim"] = res["ssim"]
                his["psnr"] = res["psnr"]
                text += f', MSE: {res["mse"]:.4f}, SSIM: {res["ssim"]:.4f}, PSNR: {res["psnr"]:.4f}'
            if epoch == 0:
                his.update(train_info)  # 记录超参数等训练信息
            history.append(his)
            logger.info(text)

    except KeyboardInterrupt:
        if isinstance(decoder, nn.DataParallel):
            decoder.module.save_model()
        else:
            decoder.save_model()

    return history
