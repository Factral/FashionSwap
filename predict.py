import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from utils import *  # Asumiendo que contiene funciones necesarias como cargar el modelo
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_model(model_path, device):
    model = torch.load("./results/" + os.path.join(model_path + '.pth'), map_location=device)
    model.eval()
    return model

def predict_image(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    image = transform(image=np.array(image))['image']
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        return output

def save_prediction(output, save_path):
    plt.imshow(output.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, required=True, help='Ruta de la imagen PNG para hacer la predicción')
parser.add_argument('--model_path', type=str, default='normalunet2_best_model', help='Ruta del modelo')
parser.add_argument('--save_dir', type=str, default='predictions', help='Ruta donde guardar la imagen de salida')
args = parser.parse_args()

device = 'cpu'
model = load_model(args.model_path, device)

output = predict_image(model, args.image_path, device)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


args.image_path = args.image_path[:-4]

save_path = args.save_dir + "/" + os.path.basename(args.image_path) + '_prediction.png'
save_prediction(output, save_path)
