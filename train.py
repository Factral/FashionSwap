import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import argparse
from torch.utils.data import Dataset, DataLoader
from dataloader import SegmentationDataset
import random
from models.unet_model import UNet
#from models.unet_resnet import UNetWithResnet50Encoder

import albumentations as A
from albumentations.pytorch import ToTensorV2
from metrics import Metrics
from utils import *
from losses import TVLoss
#import segmentation_models_pytorch as smp
from torchinfo import summary
#from dice_score import dice_loss
from datasets.load import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--exp_name', type=str, default="prueba_multiplicacion")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.001)  
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--save_dir', type=str, default='results')
parser.add_argument('--device', type=int, default=0, help='Número de GPU a usar (por defecto: 0)')
parser.add_argument('--TVweigth', type=float, default=1e-2)
parser.add_argument('--weights', type=str, default='')
parser.add_argument('--novalidation', default=False, action='store_true')

args = parser.parse_args()
wandb.init(project="fashionswap", entity="deepbeauty", name=args.exp_name)
wandb.config.update(args)

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

model_unet = UNet(3,1)
model_unet.to(device)

aug_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    A.Rotate(limit=[10, 60], p=0.4, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

aug_transform_test = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset_clean = load_from_disk('clean-train/train/')
dataset_clean = dataset_clean.with_format('np')

dataset_train_test = dataset_clean.train_test_split(test_size=0.2, seed=42)
dataset_train = dataset_train_test['train']
dataset_test = dataset_train_test['test']

dataset_train = SegmentationDataset(dataset_train, aug_transform)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

if not args.novalidation:
    dataset_test = SegmentationDataset(dataset_test,  aug_transform_test)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

optimizer = torch.optim.Adam(model_unet.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.1, verbose=True)

lossCrossE = nn.CrossEntropyLoss()

#lossTV = TVLoss()
#lossDICE = smp.losses.DiceLoss('multiclass')

metrics = Metrics()
def train(model, data_loader, optimizer, lossCrossE):
    model_unet.train()
    acc = []
    ce = []
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = lossCrossE(outputs, labels) 

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(data_loader.dataset)

        (acc_, ce_) = metrics.all_metrics(outputs, labels)
        acc.append(acc_.item())
        ce.append(ce_.item())

    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    inputs[0] =  (inputs[0] - inputs[0].min()) / (inputs[0].max() - inputs[0].min())
    ax[0].imshow(inputs[0].permute(1, 2, 0).cpu())
    ax[1].imshow(get_image(labels[0]).cpu())
    ax[2].imshow(get_image(outputs[0]).cpu())

    return epoch_loss, sum(acc)/len(acc) , sum(ce)/len(ce), fig

def validate(model, data_loader, lossCrossE):
    acc = []
    ce = []
    model.eval()
    running_loss = 0.0
    
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).float()
        outputs = model(inputs)
        
        loss = lossCrossE(outputs, labels) 

        running_loss += loss.item() * inputs.size(0)
        acc_, ce_ = metrics.all_metrics(outputs, labels)
        acc.append(acc_.item())
        ce.append(ce_.item())

    val_loss = running_loss / len(data_loader.dataset)
    scheduler.step(val_loss)

    print(labels.shape)
    print(labels.min(), labels.max())
    print(outputs.shape)
    print(outputs.min(), outputs.max())

    fig, ax = plt.subplots(1, 3, figsize=(30,10))
    inputs[0] =  (inputs[0] - inputs[0].min()) / (inputs[0].max() - inputs[0].min())
    ax[0].imshow(inputs[0].permute(1, 2, 0).cpu())
    ax[1].imshow(get_image(labels[0]).cpu())
    ax[2].imshow(get_image(outputs[0]).cpu())
    
    return val_loss, sum(acc)/len(acc) , sum(ce)/len(ce), fig

best_val_ce = 1000
#lr_change_epochs = {70: 0.1, 100: 0.1}

for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}\n-------------------------------')
    epoch_loss, train_acc, train_ce, figTrain  = train(model_unet, data_loader_train, optimizer, lossCrossE)
    
    if not args.novalidation:
        val_loss, test_acc, test_ce, fig = validate(model_unet, data_loader_test, lossCrossE)

        if test_ce < best_val_ce:
            best_val_ce = test_ce
            torch.save(model_unet, os.path.join(args.save_dir, args.exp_name+'_best_model.pth'))
            torch.save(model_unet.state_dict(), os.path.join(args.save_dir, args.exp_name+'_best_weights.pth'))
            print("Best model saved with test CE: ", best_val_ce)
    
    if args.novalidation:
        print(f'Epoch {epoch} train loss: {epoch_loss:.4f}')
    else:
        print(f'Epoch {epoch} train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}')

    if args.novalidation:
        wandb.log({'epochs': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_loss,
                'train_acc': train_acc, 'train_ce': train_ce, 'fig': figTrain})
    else:
        wandb.log({'epochs': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'train_loss': epoch_loss, 'val_loss': val_loss,
                'train_acc': train_acc, 'train_ce': train_ce,
                'test_acc': test_acc, 'test_ce': test_ce,
                'fig': fig})

    if not os.path.exists(os.path.join(args.save_dir)):
        os.makedirs(os.path.join(args.save_dir))