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
from datasets.load import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='primerintento')
parser.add_argument('--save_dir', type=str, default='predictions')
parser.add_argument('--batch_size', type=int, default=16)
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

dataset_clean = load_from_disk('clean-train/train/')
dataset_clean = dataset_clean.with_format('np')

dataset_train_test = dataset_clean.train_test_split(test_size=0.2, seed=42)
dataset_test = dataset_train_test['test']

dataset_test = SegmentationDataset(dataset_test,  aug_transform_test)
data_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')


model = load_model(os.path.join('results/', args.model_path + '.pth'), device=device)

acc = []
ce = []

metrics = Metrics()
save_path = args.save_dir+ '/'+ args.model_path+ '/'

if os.path.exists(save_path) == False:
    os.mkdir(save_path)

ido = 0
idi = 0

for i, (inputs, mask) in enumerate(data_loader):
    inputs = inputs.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        
        ido = save_images(outputs, save_path, ido, "output")
        #save_images(mask, save_path, i, "mask")
        idi = save_images(inputs, save_path, idi, "input")
        
        (acc_, ce_) = metrics.all_metrics(outputs, mask)
        acc.append(acc_.item())
        ce.append(ce_.item())
    
print(f'Accuracy: {sum(acc)/len(acc)}')
print(f'Cross Entropy: {sum(ce)/len(ce)}')

