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
        pass

    def accuracy(self, outputs, mask):
        acc = (outputs.cpu() == mask.cpu()).float().mean()
        return acc
    
    def cross_entropy(self, outputs, mask):
        ce = F.binary_cross_entropy(outputs.cpu(), mask.cpu())
        return ce
    
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
