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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

model.eval()
samples = []
count = 0
max_samples = 10

def visualize_gradcam(model, test_loader, device, target_class, real_class, max_samples=10, save_name="gradcam.png"):
    model.eval()
    samples = []
    count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            if count >= max_samples:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            mask = (preds == target_class) & (labels == real_class)
            filtered_images = images[mask]

            for img in filtered_images:
                if count >= max_samples:
                    break
                samples.append(img)
                count += 1

    target_layers = [model.convnext.stages[3].blocks[1].conv_dw]
    targets = [ClassifierOutputTarget(target_class)]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, img in enumerate(samples):
        image_tensor = img[None, :, :, :]
        image_tensor = image_tensor.to(device)

        def tensor_to_rgb(tensor):
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = tensor.cpu().numpy()
            image = image.transpose(1, 2, 0)  # CxHxW to HxWxC
            image = std * image + mean  # Denormalize
            image = np.clip(image, 0, 1)  # Clip to [0, 1]
            return image

        with GradCAM(model=model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]

            visualization = show_cam_on_image(tensor_to_rgb(image_tensor.squeeze()), grayscale_cam, use_rgb=True)

            axes[i].imshow(visualization)
            axes[i].axis('off')
            axes[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.savefig(IMG_SAVE_DIR+save_name)

print("This code will apply GradCAM to an image with the real class that you choose and the 'features' that the classification model sees of the images to a specific prediction class.")
print("So let's say you want to see which features does the model see when Plastic objects is predicted as Plastic, you enter the same class label (int)")
print("But, if you want to see which features does the model see when Plastic objects is predicted as Non-Plastic, you enter different class label (int)")
print("You can see the confusion matrix as guidance on what to see")

real_class = eval(input("Enter real class label: "))
predicted_class = eval(input("Enter predicted class label: "))
max_samples = eval(input("Enter how many samples you want to choose: "))

visualize_gradcam(model, test_loader, device, predicted_class, real_class, max_samples)