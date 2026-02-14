import math

import torch.nn as nn
import random
import numpy as np
from torch.nn.grad import conv2d_weight
import torch.nn.functional as F
import torch
def add_high_freq_noise_to_grad(grad, beta=0.1, ratio=0.5):
    """
    grad: torch.Tensor, shape [H, W]，单通道二维梯度
    beta: 扰动强度系数
    ratio: 高频掩码半径比例，越大只保留更外围的高频
    """
    H, W = grad.shape
    device = grad.device

    # 1. 原始梯度做FFT
    freq = torch.fft.fft2(grad)

    # 2. 构造高频掩码
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    center_y, center_x = H // 2, W // 2
    dist = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_dist = dist.max()
    mask = (dist >= ratio * max_dist).float()

    # 3. 生成随机复数噪声（高频）
    real = torch.randn(H, W, device=device)
    imag = torch.randn(H, W, device=device)
    noise = torch.complex(real, imag) * mask

    # 4. 高频噪声叠加到频谱
    freq_noisy = freq + noise * beta

    # 5. IFFT回到空间域，取实部
    grad_noisy = torch.fft.ifft2(freq_noisy).real

    return grad_noisy


def add_high_freq_noise_to_weight(shape, beta=0.01, ratio=0.5):
    """
    生成高频扰动：傅里叶变换 -> 保留高频区域 -> 逆变换
    输入: shape = [C_in, H, W]（每个通道的参数维度）
    """
    import torch.fft

    noise = torch.randn(shape)  # 空间域白噪声
    fft_noise = torch.fft.fft2(noise)  # 傅里叶变换到频域
    fft_shifted = torch.fft.fftshift(fft_noise)  # 中心化频谱

    _, H, W = shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    radius = ratio * min(H, W) / 2
    dist = ((X - center_x) ** 2 + (Y - center_y) ** 2).sqrt()
    mask = (dist >= radius).float()  # 高频掩码（浮点型）

    # 应用掩码，保留高频部分，保留复数形式
    high_freq_noise = fft_shifted * mask # 有广播机制

    return high_freq_noise * beta  # 返回频域的复数噪声（带缩放）



def mix_grad(model, grad_dict, lr=0.01):
    # model = deepcopy(models)
    # model.to('cup')
    with torch.no_grad():
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                layer = model.get_submodule(name)
                if isinstance(model.get_submodule(name), nn.Conv2d):
                    if layer.weight.grad is None:
                        continue  # 跳过没有梯度的参数

                    # 判断是否是卷积层的权重，并在字典中有替代梯度
                    if isinstance(model.get_submodule(name), nn.Conv2d) and name in grad_dict:
                        # 用自定义梯度替换
                        custom_grad = grad_dict[name].to(layer.weight.device)
                        layer.weight.data -= lr * custom_grad
                    else:
                        # 使用反向传播后的默认梯度
                        layer.weight.data -= lr * layer.weight.grad


def clip_feature_grad(feature_grad, clip_norm):
    # 按样本进行feature grad剪裁
    B = feature_grad.shape[0]
    grad_flat = feature_grad.view(B, -1)
    grad_norm = torch.norm(grad_flat, p=2, dim=1, keepdim=True)  # [B,1]

    # 计算缩放比例，不超过1
    clip_coef = (clip_norm / (grad_norm + 1e-6)).clamp(max=1.0)

    # 只有范数超过clip_norm才会缩放，没超过保持原样
    clipped_grad = grad_flat * clip_coef

    clipped_grad = clipped_grad.view_as(feature_grad)
    return clipped_grad





