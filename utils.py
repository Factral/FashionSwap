
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

def save_images(outputs, save_dir, idx, name): 
    id = idx * outputs.shape[0]

    for j in range(outputs.shape[0]):
        if name != "input":
            image = get_image(outputs[j])
            
            im.imsave(os.path.join(save_dir, f'{name}_{id}.png'), image.cpu())
            
        else:
            image = outputs[j]
            image = image.permute(1, 2, 0)
            image = image.float().cpu().numpy()
            id += 1  
            # normalize image between 0 and 1
            image = (image - image.min()) / (image.max() - image.min())
            im.imsave(os.path.join(save_dir, f'{name}_{id}.png'), image)
        

def get_image(mask):
    
    preds = torch.argmax(mask, dim=0)
    preds = preds.float()
        
    return preds



def apply_convex_hull(outputs):

    for i in range(outputs.shape[0]):

        contours, _ = cv2.findContours(outputs[i].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = [cv2.convexHull(c) for c in contours]
        hull_mask = np.zeros_like(outputs[i])
        cv2.drawContours(hull_mask, hull, -1, (1), thickness=cv2.FILLED)

        outputs[i] = hull_mask
    
    return hull_mask


def apply_morphology(outputs, operation='opening', kernel_size=3):
    # Convert PyTorch outputs to NumPy array
    np_array = outputs.squeeze().cpu().numpy()
    
    # Define the structuring element
    selem = morphology.disk(kernel_size)
    
    # Apply the specified morphological operation
    if operation == 'opening':
        np_array = morphology.opening(np_array, selem)
    elif operation == 'closing':
        np_array = morphology.closing(np_array, selem)
    # Add other operations as needed

    # Convert back to PyTorch outputs
    return torch.from_numpy(np_array).unsqueeze(0)


