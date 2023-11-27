import torch
import os
import matplotlib.image as im
from dataloader import SegmentationDataset
from torch.utils.data import Dataset, DataLoader
import argparse
import torch.nn.functional as F
import numpy as np
from metrics import Metrics
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='deepbeauty8_model')
parser.add_argument('--data_dir', type=str, default='Dataset_woDuplicates')
parser.add_argument('--save_dir', type=str, default='predictions')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=int, default=0, help='Número de GPU a usar (por defecto: 0)')

args = parser.parse_args()

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


img_ids = np.load(os.path.join('results/', 'test_ids.npy'))

aug_transform_test = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = SegmentationDataset(img_ids, args.data_dir, aug_transform=aug_transform_test)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


model = load_model(os.path.join('results/', args.model_path + '.pth'), device=device)

acc = []
ce = []

metrics = Metrics()
save_path = args.save_dir+ '/'+ args.model_path+ '/'

if os.path.exists(save_path) == False:
    os.mkdir(save_path)

for i, (inputs, mask) in enumerate(data_loader):
    inputs = inputs.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        outputs = multiply_mask(outputs, mask)

        save_images(outputs, save_path, i, "output")
        save_images(mask, save_path, i, "mask")
        save_images(inputs, save_path, i, "input")
        
        acc_, ce_ = metrics.all_metrics(outputs, mask)
        acc.append(acc_.item())
        ce.append(ce_.item())
    
print(f'Accuracy: {sum(acc)/len(acc)}')
print(f'Cross Entropy: {sum(ce)/len(ce)}')

