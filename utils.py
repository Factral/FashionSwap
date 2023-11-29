
import torch
import os
import matplotlib.image as im
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import morphology

def save_images(outputs, save_dir, id, name): 

    for j in range(outputs.shape[0]):
        if name != "input":
            image = outputs[j].detach()
            
            im.imsave(os.path.join(save_dir, f'{name}_{id}.png'), image.cpu())
            
        else:
            image = outputs[j]
            image = image.permute(1, 2, 0)
            image = image.float().cpu().numpy()
            id += 1  
            # normalize image between 0 and 1
            image = (image - image.min()) / (image.max() - image.min())
            im.imsave(os.path.join(save_dir, f'{name}_{id}.png'), image)
        id += 1
    return id
        
