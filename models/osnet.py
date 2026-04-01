# Kiến trúc lõi của OSNet.
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))
    
class ChannelGate(nn.Module):
    def __init__(self, in_c, reduction=16):
        super(ChannelGate, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, in_c, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.global_avg_pool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w
    
class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, bottleneck_reduction=4):
        super(OSBlock, self).__init__()
        mid_c = out_c // bottleneck_reduction
        self.conv1 = ConvLayer(in_c, mid_c, 1)
        self.conv2a = LightConv3x3(mid_c, mid_c)
        self.conv2b = nn.Sequential(LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c))
        self.conv2c = nn.Sequential(LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c))
        self.conv2d = nn.Sequential(LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c))
        self.gate = ChannelGate(mid_c)
        self.conv3 = ConvLayer(mid_c, out_c, 1)
        self.downsample = ConvLayer(in_c, out_c, 1) if in_c != out_c else None

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a + x2b + x2c + x2d)
        x3 = self.conv3(x2)
        return x3 + residual

class OSNet(nn.Module):
    def __init__(self, blocks, layers, channels, feature_dim=512):
        super(OSNet, self).__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, 2, 3)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1])
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2])
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3])
        self.conv5 = ConvLayer(channels[3], channels[3], 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels[3], feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True))
    def _make_layer(self, block, layer_num, in_c, out_c):
        layers = [block(in_c, out_c)]
        for _ in range(1, layer_num):
            layers.append(block(out_c, out_c))
        layers.append(nn.Sequential(ConvLayer(out_c, out_c, 1), nn.AvgPool2d(2, 2)))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
def osnet_x0_25():
    return OSNet(blocks=[OSBlock, OSBlock, OSBlock], layers=[2, 2, 2], channels=[16, 64, 96, 128], feature_dim=512)
