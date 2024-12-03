import os
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
from tqdm.notebook import trange, tqdm
import pickle

from commons.utils import *
from commons.config import *
from commons.dataloader import create_dataloader_cross_validation
from models.model import ConvNextV2

print(f'PyTorch version {torch.__version__}')
    
seed_everything(12)

with open("images.pickle", 'rb') as f: 
    images = pickle.loads(f.read())

with open("labels.pickle", 'rb') as f: 
    labels = pickle.loads(f.read())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using {device} device')

df = pd.DataFrame({'images': images, 'label': labels})

df_train, df_test = train_test_split(df, test_size=0.2)

# Cross-validation
folds = StratifiedKFold(n_splits=FOLD_NUM, shuffle=True, random_state=SEED)\
        .split(np.arange(df_train.shape[0]), df_train['label'].to_numpy())

# For Visualization
train_acc_list = []
valid_acc_list = []
train_loss_list = []
valid_loss_list = []


for fold, (trn_idx, val_idx) in enumerate(folds):
    print(f'==========Cross Validation Fold {fold+1}==========')
    # Load Data
    train_loader, valid_loader = create_dataloader_cross_validation(df, trn_idx, val_idx)

    # Load model, loss function, and optimizing algorithm
    model = ConvNextV2(NUM_CLASSES).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
            
    # For Visualization
    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []

    # Start training
    best_acc = 0
    for epoch in range(EPOCHS_CROSSVAL):
        time_start = time.time()
        print(f'==========Epoch {epoch+1} Start Training==========')
        model.train()
        
        epoch_loss = 0
        epoch_accuracy = 0
    
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, (img, label) in pbar:
            img = img.to(device).float()
            label = label.to(device).long()

            output = model(img)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        print(f'==========Epoch {epoch+1} Start Validation==========')
        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            val_labels = []
            val_preds = []

            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for step, (img, label) in pbar:
                img = img.to(device).float()
                label = label.to(device).long()

                val_output = model(img)
                val_loss = loss_fn(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

                val_labels += [label.detach().cpu().numpy()]
                val_preds += [torch.argmax(val_output, 1).detach().cpu().numpy()]
            
            val_labels = np.concatenate(val_labels)
            val_preds = np.concatenate(val_preds)
        
        exec_t = int((time.time() - time_start)/60)
        print(
            f'Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f} / Exec time {exec_t} min\n'
        )

        train_accs.append(epoch_accuracy.cpu().numpy())
        valid_accs.append(epoch_val_accuracy.cpu().numpy())
        train_losses.append(epoch_loss.detach().cpu().numpy())
        valid_losses.append(epoch_val_loss.detach().cpu().numpy())
    
    train_acc_list.append(train_accs)
    valid_acc_list.append(valid_accs)
    train_loss_list.append(train_losses)
    valid_loss_list.append(valid_losses)
    del model, optimizer, train_loader, valid_loader, train_accs, valid_accs, train_losses, valid_losses
    gc.collect()
    torch.cuda.empty_cache()

show_validation_score(train_acc_list, train_loss_list, valid_acc_list, valid_loss_list)