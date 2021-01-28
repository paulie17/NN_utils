import torch
import torch.nn as nn
from efficientnet_pytorch.utils import Conv2dDynamicSamePadding,MaxPool2dDynamicSamePadding

def alexnet(pretrained = True):
  repo = 'pytorch/vision'
  return torch.hub.load(repo, 'alexnet', pretrained)

def conv_block_layers(in_channels, out_channels, ks=3, stride=1, pool = False, activation = True):
  layers = [Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size=ks, stride=stride, bias = False),
            nn.BatchNorm2d(out_channels)]
  if activation: layers.append(nn.ReLU(inplace = True))
  if pool: layers.append(MaxPool2dDynamicSamePadding(kernel_size=3, stride=stride))
  return layers

def bn_relu_conv(in_channels, out_channels, ks = 3, stride = 1):
  layers = [nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            Conv2dDynamicSamePadding(in_channels,out_channels,kernel_size=ks, stride = stride, bias = False)
  ]
  return layers

def classification_layers(out_size, num_classes):
  layers = [nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(out_size, num_classes)
  ]
  return layers

class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self,  ni, nf, stride=1, is_first_block_of_first_layer=False):
    super().__init__()

    if is_first_block_of_first_layer:
    # don't repeat bn->relu since we just did bn->relu->maxpool
      self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(ni, nf, kernel_size=3, stride=stride))
    else:
      self.conv1 = nn.Sequential(*bn_relu_conv(ni,nf,stride=stride))


    self.residual = nn.Sequential(*bn_relu_conv(nf,nf*BasicBlock.expansion))

    #shortcut
    self.shortcut = nn.Sequential()

    #if the shortcut output dimension is not the same as the residual output
    #dimension, match them with 1x1 convolution
    if stride != 1 or ni != BasicBlock.expansion*nf:
      self.shortcut = nn.Sequential(
          nn.Conv2d(ni,nf*BasicBlock.expansion,kernel_size=1, stride = stride, bias = False )
      )

  def forward(self,x):
    out = self.conv1(x)
    out = self.residual(out)
    return out + self.shortcut(x)

class BottleNeck(nn.Module):

  expansion = 4

  def __init__(self,  ni, nf, stride = 1, is_first_block_of_first_layer=False):
    super().__init__()

    if is_first_block_of_first_layer:
    # don't repeat bn->relu since we just did bn->relu->maxpool
      self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(ni, nf, kernel_size=1, stride=stride))
    else:
      self.conv1 = nn.Sequential(*bn_relu_conv(ni,nf,stride=stride))

    self.conv2 = nn.Sequential(*bn_relu_conv(nf, nf, ks=3))
    self.residual = nn.Sequential(*bn_relu_conv(nf, nf*BottleNeck.expansion,ks =1))


    self.shortcut = nn.Sequential()

    #self.shortcut_stride = int(round(x.shape[2] / out.shape[2]))

    if  stride != 1 or ni != nf*BottleNeck.expansion:
      self.shortcut = nn.Sequential(
          nn.Conv2d(ni,nf*BottleNeck.expansion,kernel_size=1, stride = stride, bias = False )
      )

  def forward(self,x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.residual(out)
    return out + self.shortcut(x)


class ResNet(nn.Module):

  def __init__(self,block,num_block,num_classes=1000):
    super().__init__()

    self.in_channels = 64

    self.init =  nn.Sequential(*conv_block_layers(3,self.in_channels,ks=7,stride=2, pool=True))
    self.layer1 = self._make_layer(block, 64,  num_block[0], 1, is_first_layer = True)
    self.layer2 = self._make_layer(block, 128,  num_block[1], 2)
    self.layer3 = self._make_layer(block, 256,  num_block[2], 2)
    self.layer4 = self._make_layer(block, 512,  num_block[3], 2)
    self.postnorm = nn.BatchNorm2d(512*block.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.classification =  nn.Sequential(*classification_layers(512*block.expansion, num_classes))

  def _make_layer(self, block, out_channels, num_blocks, stride, is_first_layer = False):

    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for i,stride in enumerate(strides):
      layers.append(block(self.in_channels,out_channels,stride,is_first_block_of_first_layer=(is_first_layer and i == 0)))
      self.in_channels = out_channels * block.expansion

    return nn.Sequential(*layers)

  def forward(self, x):

        out = self.init(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.postnorm(out)
        out = self.relu(out)
        out = self.classification(out)
        return out

def Resnet18(num_classes = 25):
  return ResNet(BasicBlock, [2, 2, 2, 2], num_classes= num_classes)

def Resnet34(num_classes = 25):
  return ResNet(BasicBlock, [3, 4, 6, 3], num_classes= num_classes)

def Resnet50(num_classes = 25):
  return ResNet(BottleNeck, [3, 4, 6, 3], num_classes= num_classes)
