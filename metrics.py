import torch
import os
import matplotlib.image as im
from dataloader import SegmentationDataset
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn.functional as F
import numpy as np

class Metrics:  
    def __init__(self):
        self.acc = 0
        self.ce = 0

    def one_hot(self, outputs):   
        max_indices = torch.argmax(outputs, dim=1)
        one_hot = F.one_hot(max_indices, num_classes=outputs.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        return one_hot

    def accuracy(self, outputs, mask):
        one_hot = self.one_hot(outputs)
        self.acc = (one_hot.cpu() == mask.cpu()).float().mean()
        return self.acc
    
    def cross_entropy(self, outputs, mask):
        self.ce = F.cross_entropy(outputs.cpu(), mask.cpu())
        return self.ce
    
    def mse(self, outputs, mask):
        mse = F.mse_loss(outputs, mask)
        return mse

    def all_metrics(self, outputs, mask, reconstruction=False):
        if reconstruction:
            mse = self.mse(outputs, mask)
            return mse
        acc = self.accuracy(outputs, mask)
        ce = self.cross_entropy(outputs, mask)
        return acc, ce
