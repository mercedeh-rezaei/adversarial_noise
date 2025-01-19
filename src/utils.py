from PIL import Image
import torchvision.transforms as transforms
import torch
import os

def load_image(image):
    """
    Loading and preprocessing the image
    """
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

def save_image(tensor, output):
    """Saves a tensor as an output"""

    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    unnormalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    image = tensor.clone().squeeze(0) 
    image = unnormalize(image).clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(output)

def denorm(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalizes a batch of images.
    """
    mean = torch.tensor(mean).to(batch.device)
    std = torch.tensor(std).to(batch.device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)