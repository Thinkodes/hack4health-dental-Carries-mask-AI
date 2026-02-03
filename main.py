# ==========================
# Name: Ansh Mathur
# github: https://github.com/Thinkodes
# ==========================

import sys
sys.path.append("..")

# This is a private Library, Please Contact Ansh mathur, am3274@srmist.edu.in to gain access. A* Conference
# (ICML2026) submission, review pending.
from titan import Linear, Dense, Model

import os
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np


ROOT = os.path.abspath("Dataset")
CLASSES = ["Carries", "Normal"]
IMAGE_SIZE = (256, 256)  
INPUT_SIZE = 256 * 256  
OUTPUT_SIZE = 256 * 256  

transform = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.Grayscale(),
    T.ToTensor(),  
])

X_list = []
Y_list = []

for label, cls in enumerate(CLASSES):
    cls_dir = os.path.join(ROOT, cls)
    
    if not os.path.exists(cls_dir):
        print(f"Warning: Directory {cls_dir} not found!")
        continue
    
    for fname in os.listdir(cls_dir):
        
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")) or fname.endswith("-mask.png"):
            continue

        path = os.path.join(cls_dir, fname)
        
        
        img = Image.open(path)
        img_t = transform(img)  
        img_t = img_t.flatten().unsqueeze(0)  
        X_list.append(img_t)
        
        
        if "-mask" in fname:
            
            mask_t = transform(img).flatten().unsqueeze(0)  
        else:
            
            base_name = os.path.splitext(fname)[0]
            mask_path = os.path.join(cls_dir, f"{base_name}-mask.png")
            
            if os.path.exists(mask_path):
                
                mask_img = Image.open(mask_path)
                mask_t = transform(mask_img).flatten().unsqueeze(0)  
            else:
                
                mask_t = torch.zeros(1, INPUT_SIZE)
        
        Y_list.append(mask_t)

if len(X_list) == 0:
    raise ValueError("No images found in dataset directories!")

X = torch.cat(X_list, dim=0).float()
Y = torch.cat(Y_list, dim=0).float()  

print("X shape:", X.shape)
print("Y shape (masks):", Y.shape)
print(f"Dataset size: {X.shape[0]} samples")
print(f"Mask values range: [{Y.min():.3f}, {Y.max():.3f}]")

model = Model(
    Dense(1024, 1024),
    Linear(1024, 65536)
)

print("Model architecture:")
print(model)

print("\nTraining model for mask prediction...")

model.fit(X, Y)


print("\nEvaluating mask prediction...")


predictions = model(X)

mse = torch.nn.functional.mse_loss(predictions, Y)
mae = torch.nn.functional.l1_loss(predictions, Y)
ssim_score = 1 - mse / (torch.var(Y) + 1e-8)  

print(f"MSE: {mse.item():.6f}")
print(f"MAE: {mae.item():.6f}")
print(f"SSIM (approx): {ssim_score.item():.6f}")

os.makedirs("predictions", exist_ok=True)

for i in range(min(10, X.shape[0])):  
    
    pred_mask = predictions[i].reshape(256, 256).detach().numpy()
    true_mask = Y[i].reshape(256, 256).detach().numpy()
    
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    
    orig_img = X[i].reshape(256, 256).detach().numpy()
    axes[0].imshow(orig_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Predicted Output Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"predictions/sample_{i}.png", dpi=150, bbox_inches='tight')
    plt.close()