import os
import torch
import torchvision
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as tt
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking = True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def download_full_dataset():
    data_url = "http://www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz"
    download_url(data_url,'.')
    extract_archive('./wikipaintings_full.tgz','./data')
    data_dir = './data/wikipaintings_full'
    classes = os.listdir(data_dir + "/wikipaintings_train")
    return data_dir, classes


def download_small_dataset():
    dataset_path = './drive/MyDrive/NN_project/wikipaintings_small.tgz'
    extract_archive(dataset_path,'./data')
    data_dir = './data/wikipaintings_small'
    classes = os.listdir(data_dir + "/wikipaintings_train")
    return data_dir, classes

def train_valid_test_dataset(data_dir):
    train_tfms = tt.Compose([
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation((0,90)),
                         tt.ToTensor(),
                         tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         tt.RandomResizedCrop(size = (224), scale=(0.5,1))
                         ])
    valid_tfms = tt.Compose([
                         SquarePad(),
                         tt.Resize((224,224)),
                         tt.ToTensor(),
                         tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                         ])
    train_ds = ImageFolder(data_dir + '/wikipaintings_train', train_tfms)
    valid_ds = ImageFolder(data_dir + '/wikipaintings_val',valid_tfms)
    test_ds = ImageFolder(data_dir + '/wikipaintings_test',valid_tfms)
    return train_ds, valid_ds, test_ds

class SquarePad:
  def __call__(self, image):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    pad = tt.Pad(padding)
    return pad(image)

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
    for images,labels in dl:
      fig,ax = plt.subplots(figsize=(12,12))
      ax.set_xticks([]); ax.set_yticks([])
      denorm_images = denormalize(images)
      ax.imshow(make_grid(denorm_images, nrow=nrow).permute(1,2,0).clamp(0.1))
      break
