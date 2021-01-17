import os
import torch
import torchvision
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt


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
