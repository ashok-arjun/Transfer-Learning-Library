'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AutoGrowCifarPlainNet', 'AutoGrowCifarResNetBasic', 'AutoGrowCifarPlainNoBNNet']


class PlainBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return out

class PlainNoBNBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PlainNoBNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CifarResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, image_channels=3, batchnorm=True):
        super(CifarResNet, self).__init__()
        self.block = block
        self.in_planes = 16
        self.out_planes = 64
        if batchnorm:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=True)
            self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.out_features = 64*block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

def AutoGrowCifarResNetBasic(num_blocks, num_classes=10, image_channels=3):
    assert len(num_blocks) == 3, "3 blocks are needed, but %d is found." % len(num_blocks)
    print ('num_classes=%d, image_channels=%d' % (num_classes, image_channels))
    return CifarResNet(BasicBlock, num_blocks, num_classes=num_classes, image_channels=image_channels)

def AutoGrowCifarPlainNet(num_blocks, num_classes=10, image_channels=3):
    assert len(num_blocks) == 3, "3 blocks are needed, but %d is found." % len(num_blocks)
    print ('num_classes=%d, image_channels=%d' % (num_classes, image_channels))
    # CifarResNet is NOT a ResNet, it's just a building func
    return CifarResNet(PlainBlock, num_blocks, num_classes=num_classes, image_channels=image_channels)

def AutoGrowCifarPlainNoBNNet(num_blocks, num_classes=10, image_channels=3):
    assert len(num_blocks) == 3, "3 blocks are needed, but %d is found." % len(num_blocks)
    print ('num_classes=%d, image_channels=%d' % (num_classes, image_channels))
    # CifarResNet is NOT a ResNet, it's just a building func
    return CifarResNet(PlainNoBNBlock, num_blocks, num_classes=num_classes, image_channels=image_channels, batchnorm=False)
