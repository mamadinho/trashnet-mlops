from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from .dataset import CustomDataset
from .config import *

def create_dataloader_cross_validation(df, trn_idx, val_idx):
    images = df['images'].to_list()
    labels = df['label'].to_numpy()

    # Define transformations for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Subset(CustomDataset(images, labels, transform=train_transform), trn_idx)
    valid_dataset = Subset(CustomDataset(images, labels, transform=val_transform), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, valid_loader

def create_dataloader_full_train(images, labels, transform, batch_size, shuffle=True):
    dataset = CustomDataset(images, labels, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS)