import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from PIL import Image
import timm

import os
import numpy as np
import pandas as pd
import math
import time
import random
import gc
import cv2
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import copy

# Image augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Modeling
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from tqdm import tqdm
import pickle

from commons.utils import *
from commons.config import *
from commons.dataloader import create_dataloader_full_train
from models.model import ConvNextV2
import wandb

print(f'PyTorch version {torch.__version__}')
print(f'Albumentations version {A.__version__}')
    
seed_everything(12)

with open("images.pickle", 'rb') as f: 
    images = pickle.loads(f.read())

with open("labels.pickle", 'rb') as f: 
    labels = pickle.loads(f.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

df = pd.DataFrame({'images': images, 'label': labels})

df_train, df_test = train_test_split(df, test_size=0.2)

df_train_final, df_val_final = train_test_split(df_train, test_size=0.2)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_loader = create_dataloader_full_train(
    df_train_final['images'].to_list(),
    df_train_final['label'].to_numpy(),
    train_transform,
    batch_size=BATCH_SIZE
)

val_loader = create_dataloader_full_train(
    df_val_final['images'].to_list(),
    df_val_final['label'].to_numpy(),
    val_transform,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if USE_WANDB:
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(
        project=WANDB_PROJECT,
        config={
        "learning_rate": LR,
        "architecture": ARCHITECTURE,
        "dataset": WANDB_DATASET,
        "epochs": EPOCHS_FULL,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.1
        }
    )

model = ConvNextV2(NUM_CLASSES).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

early_stopping_patience = 10
best_val_acc = 0
early_stopping_counter = 0

train_acc_list, train_loss_list = [], []
val_acc_list, val_loss_list = [], []
train_f1_list, val_f1_list = [], []  # For storing F1 scores

best_model = None

for epoch in range(EPOCHS_FULL):
    time_start = time.time()
    print(f'========== Epoch {epoch+1} Start Training ==========') 
    model.train()
    
    epoch_loss = 0
    epoch_accuracy = 0
    all_preds, all_labels = [], []  # For F1-score and confusion matrix calculation in training
    
    # Training loop
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (img, label) in pbar:
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(img)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        # Collect predictions for F1-score and confusion matrix calculation
        _, preds = output.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        if USE_WANDB:
            wandb.log({"train_batch_loss": loss,
                    "train_batch_acc": acc})

    # Calculate F1-score for the training set
    train_f1 = f1_score(all_labels, all_preds, average='weighted')
    train_f1_list.append(train_f1)
    
    # Calculate confusion matrix for the training set
    if USE_WANDB:
        wandb.log({"train_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                            y_true=all_labels, preds=all_preds,
                                                            class_names=LABELS_NAME, title="Training Confusion Matrix")})

    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_accuracy = 0
        val_preds, val_labels = [], []  # For F1-score and confusion matrix calculation in validation
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = loss_fn(output, label)

            val_loss += loss / len(val_loader)
            acc = (output.argmax(dim=1) == label).float().mean()
            val_accuracy += acc / len(val_loader)

            # Collect predictions for F1-score and confusion matrix calculation
            _, preds = output.max(1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    # Calculate F1-score for the validation set
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_f1_list.append(val_f1)

    # Calculate confusion matrix for the validation set
    if USE_WANDB:
        wandb.log({"val_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                            y_true=val_labels, preds=val_preds,
                                                            class_names=LABELS_NAME, title="Validation Confusion Matrix")})

    # Log results
    train_acc_list.append(epoch_accuracy.detach().cpu().numpy())
    train_loss_list.append(epoch_loss.detach().cpu().numpy())
    val_acc_list.append(val_accuracy.detach().cpu().numpy())
    val_loss_list.append(val_loss.detach().cpu().numpy())

    scheduler.step(val_loss)  # Update learning rate scheduler

    exec_t = int((time.time() - time_start) / 60)
    print(f"Epoch: {epoch+1} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_accuracy:.4f} - Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f} - Val F1: {val_f1:.4f} / Exec Time: {exec_t} min\n")
    
    # Log metrics to W&B
    if USE_WANDB:
        wandb.log({"train_loss": epoch_loss, 
                "train_acc": epoch_accuracy, 
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "lr": scheduler.get_last_lr()[-1],
                "epoch": epoch})

    # Early stopping
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        early_stopping_counter = 0
        print("Validation accuracy improved, saving model.")
        torch.save(model.state_dict(), f'model_lr_{LR}_bs_{BATCH_SIZE}.pth')
        best_model = copy.deepcopy(model)
    else:
        early_stopping_counter += 1
        print(f"No improvement in validation accuracy for {early_stopping_counter} epochs.")

    if early_stopping_counter >= early_stopping_patience:
        print("Early stopping triggered.")
        break

if USE_WANDB:
    wandb.log_artifact(f"model_lr_{LR}_bs_{BATCH_SIZE}.pth", name=f'model_lr_{LR}_bs_{BATCH_SIZE}', type='model') 

test_loader = create_dataloader_full_train(
    df_test['images'].to_list(),
    df_test['label'].to_numpy(),
    val_transform,
    batch_size=BATCH_SIZE
)

model = best_model

all_preds = []
all_labels = []
total_loss = 0
correct_predictions = 0
total_samples = 0
class_names = LABELS_NAME

for img, label in test_loader:
    img, label = img.to(device), label.to(device)
    
    output = model(img)
    
    loss = F.cross_entropy(output, label)
    total_loss += loss.item()

    _, preds = torch.max(output, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(label.cpu().numpy())

    correct_predictions += (preds == label).sum().item()
    total_samples += label.size(0)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = correct_predictions / total_samples

f1 = f1_score(all_labels, all_preds, average='weighted')  # weighted F1-score

conf_mat = confusion_matrix(all_labels, all_preds)

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test F1-Score: {f1:.4f}')
print(f'Test Loss: {total_loss / len(test_loader):.4f}')

wandb.run.summary["best_val_acc"] = best_val_acc
wandb.run.summary["test_acc"] = accuracy
wandb.run.summary["test_f1"] = f1
wandb.run.summary["test_loss"] = total_loss

show_misclassified_samples(conf_mat, LABELS_NAME)

wandb.finish()
