import torch
from torch import nn

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def get_weight(self):
        w = self.conv.weight
        o, i, k1, k2 = w.shape
        w_flat = w.reshape(o, i, k1 * k2)
        w_cd = torch.zeros_like(w_flat, device=w.device, dtype=w.dtype)
        w_cd[:, :, :] = w_flat
        w_cd[:, :, 4] = w_flat[:, :, 4] - w_flat.sum(dim=2)
        w_cd = w_cd.reshape(o, i, k1, k2)
        return w_cd, self.conv.bias

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)

    def forward(self, x):
        x = x.contiguous()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        w, b = self.conv1_1.get_weight()
        res = nn.functional.conv2d(input=attn, weight=w, bias=b, stride=1, padding=1, groups=1)
        return x * res + x
