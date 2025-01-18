import torch
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from attack import fgsm_attack
from utils import load_image, save_image
import argparse
import matplotlib.pyplot as plt
import os 
import urllib
import json
from PIL import Image


def load_imagenet_classes():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        urllib.request.urlretrieve(url, "imagenet_class_index.json")
        with open("imagenet_class_index.json") as f:
            class_idx = json.load(f)
        return {int(key): value[1] for key,value in class_idx.items()}
    except Exception as e:
        print(f"Error loading ImageNet class labels: {e}")
        return {}
    
imagenet_classes = load_imagenet_classes()


def download_image(url, filename):
    """Download image from a URL"""
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)
    print(f"Image downloaded: {filename}")


def main():

    if torch.cuda.is_available():
        device ="cuda"
    elif torch.backends.mps.is_available():
        device ="mps"
    else:
        device = "cpu"
    print(f"Using device {device}")

    parser = argparse.ArgumentParser(description="Adversarial Noise")
    parser.add_argument("target_class", type=int, help="desired target class for tampering with classification")
    # parser.add_argument("output_path", type=str, default="examples/output.jpg", help="Path to save the perturbed image")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation strength")
    args = parser.parse_args()

    # show options to choose from
    print("Here are some of the available ImageNet classes:")
    for i in range(10):  # only 10 to keep it simple
        print(f"{i}: {imagenet_classes.get(i, 'Unknown')}")

    
    url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    download_image(url, filename)

    # Load the image
    image = Image.open(filename).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    save_image(image_tensor, 'examples/orig_img.jpg')

    output_path = "examples/perturbed_image.jpg"

    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    
    # getting the original prediction
    with torch.no_grad():
        orig_pred = model(image_tensor)
        _,orig_pred_class = torch.max(orig_pred, 1)
        orig_pred_class = orig_pred_class.item()
        orig_pred_class_name = imagenet_classes.get(orig_pred_class, 'Unknown')



    target_class = args.target_class
    # performing the attack
    perturbed_image = fgsm_attack(model, image_tensor, target_class, epsilon=args.epsilon)


    # getting the prediction after attack
    with torch.no_grad():
        perturbed_pred = model(perturbed_image)
        _, perturbed_pred_class = torch.max(perturbed_pred, 1)
        perturbed_pred_class = perturbed_pred_class.item()
        perturbed_pred_class_name = imagenet_classes.get(perturbed_pred_class, 'Unknown')

    print("Top 5 predictions (original): ", torch.topk(orig_pred, 5).indices)
    print("Top 5 predictions (perturbed):", torch.topk(perturbed_pred, 5).indices)
    
    save_image(perturbed_image, output_path)
    print(f"Adversarial image saved to {output_path}")


    # Visualize the original and adversarial images
    orig_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    perturbed_img_np = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image
    axes[0].imshow(orig_img)
    axes[0].set_title(f'Original Image\nPredicted: {orig_pred_class_name}')
    axes[0].axis('off')

    # Perturbed image
    axes[1].imshow(perturbed_img_np)
    axes[1].set_title(f'Adversarial Image\nPredicted: {perturbed_pred_class_name}\nTarget: {imagenet_classes.get(target_class, "Unknown")}')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()