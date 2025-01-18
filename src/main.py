import torch
from torchvision.models import resnet18
from src.attack import fgsm_attack
from src.utils import load_image, save_image
import argparse

def main():
    parser = argparse.ArgumentParser(description="Adversarial Noise")
    parser.add_argument("image_path", type=str, help="path to the input image")
    parser.add_argument("target_class", type=int, help="desired target class for tampering with classification")
    parser.add_argument("output_path", type=str, default="examples/output.jpg")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Perturbation strength")
    args = parser.parse_args()


    model = resnet18(pretrained=True).eval()
    input_image = load_image("examples/input.jpg")

    # TODO: Implement the attack function
    perturbed_image = fgsm_attack(model, input_image, target_class=0)
    save_image(perturbed_image, "examples/output.jpg")
    print(f"Adversarial image saved to {args.output_path}")


if __name__ == "__main__":
    main()