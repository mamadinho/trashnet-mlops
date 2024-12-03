import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from torchvision import transforms
from commons.config import *
from commons.dataloader import create_dataloader_full_train
from models.model import ConvNextV2
from sklearn.model_selection import train_test_split

with open("images.pickle", 'rb') as f: 
    images = pickle.loads(f.read())

with open("labels.pickle", 'rb') as f: 
    labels = pickle.loads(f.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNextV2(n_out=6)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)

df = pd.DataFrame({'images': images, 'label': labels})
df_train, df_test = train_test_split(df, test_size=0.2)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_loader = create_dataloader_full_train(
    df_test['images'].to_list(),
    df_test['label'].to_numpy(),
    val_transform,
    batch_size=BATCH_SIZE
)

def plot_class_predictions(dataloader, device, pred_class, real_class, num_samples, save_dir=IMG_SAVE_DIR, save_name="misclassified_samples.png"):
    # Convert mean and std to numpy arrays
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    model.eval()
    samples = []
    count = 0
    
    # Collect samples
    with torch.no_grad():
        for images, labels in dataloader:
            if count >= num_samples:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # Find samples where prediction matches pred_class and true label matches real_class
            mask = (preds == pred_class) & (labels == real_class)
            filtered_images = images[mask]
            
            for img in filtered_images:
                if count >= num_samples:
                    break
                samples.append(img)
                count += 1
    
    if len(samples) == 0:
        print(f"No samples found where model predicted class {pred_class} for true class {real_class}")
        return
    
    # Create plot grid
    num_cols = min(5, num_samples)
    num_rows = (len(samples) + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3, num_rows*3))
    
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.reshape(-1, 1) if num_cols == 1 else axes.reshape(1, -1)
    
    # Plot images
    for idx, img in enumerate(samples):
        row = idx // num_cols
        col = idx % num_cols
        
        # Denormalize image
        img = img.cpu().numpy()
        img = img * std[:, None, None] + mean[:, None, None]
        img = np.clip(img.transpose(1, 2, 0), 0, 1)
        
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    
    # Turn off empty subplots
    for idx in range(len(samples), num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Predicted Class: {LABELS_NAME[pred_class]}, True Class: {LABELS_NAME[real_class]}')
    plt.tight_layout()
    plt.savefig(save_dir+save_name)

real_class = eval(input("Enter real class label: "))
predicted_class = eval(input("Enter predicted class label: "))
num_samples = eval(input("Enter how many samples you want to choose: "))

plot_class_predictions(test_loader, device, predicted_class, real_class, num_samples)
