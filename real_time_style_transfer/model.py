import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
import torch.utils.data as data
from PIL import Image

#VGG16作为特征提取网络
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures,self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = vgg[:4]   # relu1_2
        self.slice2 = vgg[4:9]  # relu2_2
        self.slice3 = vgg[9:16] # relu3_3
        self.slice4 = vgg[16:23]# relu4_3
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
    
#实际图像变换网络
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True), # 改为 InstanceNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels, affine=True), # 改为 InstanceNorm
        )

    def forward(self, x):
        return x + self.conv(x)

# 2. 图像转换网络 (Image Transformation Network)
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # 下采样：使用跨步卷积
        self.down = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # 5个残差块
        self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        
        # 上采样：使用 Upsample + Conv 替代 ConvTranspose2d 以消除棋盘效应
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, kernel_size=9, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        # 论文中通常对输出进行缩放，此处直接输出由 Tanh 引导或直接 Linear
        return self.up(self.res_blocks(self.down(x)))
    