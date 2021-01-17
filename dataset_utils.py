import os
import torch
import torchvision
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as tt

import matplotlib
import matplotlib.pyplot as plt


def download_full_dataset():
    data_url = "http://www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz"
    download_url(data_url,'.')
    extract_archive('./wikipaintings_full.tgz','./data')
    dataset_path = './wikipaintings_full.tgz'
    extract_archive(dataset_path,'./data')
    data_dir = './data/wikipaintings_full'
    classes = os.listdir(data_dir + "/wikipaintings_train")
    return data_dir, classes


def download_small_dataset():
    dataset_path = './drive/MyDrive/NN_project/wikipaintings_small.tgz'
    extract_archive(dataset_path,'./data')
    data_dir = './data/wikipaintings_small'
    classes = os.listdir(data_dir + "/wikipaintings_train")
    return data_dir, classes

def train_valid_dataset(data_dir):
    train_tfms = tt.Compose([tt.RandomHorizontalFlip(),
                         tt.RandomRotation((0,90)),
                         tt.ToTensor(),
                         tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         tt.RandomResizedCrop(size = (227), scale=(0.5,1))])
    valid_tfms = tt.Compose([tt.ToTensor(),
                         tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         tt.RandomResizedCrop(size = (227), scale=(0.5,1))])
    train_ds = ImageFolder(data_dir + '/wikipaintings_train', train_tfms)
    valid_ds = ImageFolder(data_dir + '/wikipaintings_val',valid_tfms)
    return train_ds, valid_ds

def denormalize(images, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
  means = torch.tensor(means).reshape([1,3,1,1])
  stds = torch.tensor(stds).reshape([1,3,1,1])
  return images * stds + means

#stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def show_example(dataset,idx):
    img,label = dataset[idx]
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1,2,0))

def show_batch(dl,nrow):
  stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  for images,labels in dl:
    fig,ax = plt.subplots(figsize=(12,12))
    ax.set_xticks([]); ax.set_yticks([])
    denorm_images = denormalize(images, *stats)
    ax.imshow(make_grid(denorm_images, nrow=nrow).permute(1,2,0).clamp(0.1))
    break
