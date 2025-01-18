import torch
from torchvision.models import resnet18
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from attack import fgsm_attack
from utils import load_image, save_image
import argparse
import matplotlib.pyplot as plt
import os 

def main():

    if torch.cuda.is_available():
        device ="cuda"
    elif torch.backends.mps.is_available():
        device ="mps"
    else:
        device = "cpu"
    print(f"Using device {device}")

    parser = argparse.ArgumentParser(description="Adversarial Noise")
    #parser.add_argument("target_class", type=int, help="desired target class for tampering with classification")
    parser.add_argument("output_path", type=str, default="examples/output.jpg")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation strength")
    args = parser.parse_args()

    # load the CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(train_set, batch_size=4, shuffle=True)

    if not os.path.exists('examples'):
        os.mkdir('examples')

    
    model = resnet18(pretrained=True).to(device).eval()
    model.eval()
    
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    # chossing the first image for testing
    image = images[0].unsqueeze(0).to(device)
    target_class = 3 # for example 3 refers to cat

    save_image(image, 'examples/image1.jpg')

    perturbed_image = fgsm_attack(model, image, target_class, epsilon=args.epsilon)

    save_image(perturbed_image, args.output_path)
    print(f"Adversarial image saved to {args.output_path}")

    orig_img = images[0].permute(1,2,0).numpy()

    fig, axes = plt.subplots(1,2,figsize=(12,6))
    axes[0].imshow(orig_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    perturbed_image_np = perturbed_image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    axes[1].imshow(perturbed_image_np)
    axes[1].set_title('Image with added noise')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()