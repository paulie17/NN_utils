import torch
import torch.nn as nn

# Utilities to build the architecture of the standard ResNet

def alexnet(pretrained = True):
  repo = 'pytorch/vision'
  return torch.hub.load(repo, 'alexnet', pretrained)

def conv_block_layers(in_channels, out_channels, ks=3, stride=1, pool = False, activation = True):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=ks//2, bias = False),
            nn.BatchNorm2d(out_channels)]
  if activation: layers.append(nn.ReLU(inplace = True))
  if pool: layers.append(nn.MaxPool2d(3, stride = stride, padding = 1))
  return layers

def classification_layers(out_size, num_classes):
  layers = [nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(out_size, num_classes)
  ]
  return layers

class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self,  ni, nf, stride=1):
    super().__init__()

    layers_res = conv_block_layers(ni, nf, stride = stride)
    layers_res += (conv_block_layers(nf, nf * BasicBlock.expansion, activation = False))

    self.residual = nn.Sequential(*layers_res)

    #shortcut
    self.shortcut = nn.Sequential()

    #if the shortcut output dimension is not the same as the residual output
    #dimension, match them with 1x1 convolution
    if stride != 1 or ni != BasicBlock.expansion*nf:
      self.shortcut = nn.Sequential(
          *conv_block_layers(ni,nf * BasicBlock.expansion, ks = 1, stride = stride, activation=False)
      )

  def forward(self,x):
    return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class BottleNeck(nn.Module):

  expansion = 4

  def __init__(self,  ni, nf, stride = 1):
    super().__init__()
    layers_res = conv_block_layers(ni, nf, ks=1)
    layers_res += conv_block_layers(nf, nf, stride=stride)
    layers_res += conv_block_layers(nf, nf * BottleNeck.expansion, ks=1, activation = False)

    self.residual = nn.Sequential(*layers_res)

    self.shortcut = nn.Sequential()

    if  stride != 1 or ni != nf*BottleNeck.expansion:
      self.shortcut = nn.Sequential(
          *conv_block_layers(ni, nf*BottleNeck.expansion, ks =1, stride=stride, activation=False)
      )

  def forward(self,x):
    return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))

class ResNet(nn.Module):

  def __init__(self,block,num_block,num_classes=1000):
    super().__init__()

    self.in_channels = 64

    self.resnet = nn.Sequential()
    self.resnet.add_module('init', nn.Sequential(*conv_block_layers(3,self.in_channels,ks=7,stride=2, pool=True)))
    self.resnet.add_module('layer1', self._make_layer(block, 64,  num_block[0], 1))
    self.resnet.add_module('layer2', self._make_layer(block, 128,  num_block[1], 1))
    self.resnet.add_module('layer3', self._make_layer(block, 256,  num_block[2], 2))
    self.resnet.add_module('layer4', self._make_layer(block, 512,  num_block[3], 2))
    self.resnet.add_module('classification', nn.Sequential(*classification_layers(512*block.expansion, num_classes)) )

  def _make_layer(self, block, out_channels, num_blocks, stride):

    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_channels,out_channels,stride))
      self.in_channels = out_channels * block.expansion

    return nn.Sequential(*layers)

  def forward(self,x):
    return self.resnet(x)

def resnet18(num_classes = 1000):
  return ResNet(BasicBlock, [2, 2, 2, 2], num_classes= num_classes)

def resnet34(num_classes = 1000):
  return ResNet(BasicBlock, [3, 4, 6, 3], num_classes= num_classes)

def resnet50(num_classes = 1000):
  return ResNet(BottleNeck, [3, 4, 6, 3], num_classes= num_classes)
