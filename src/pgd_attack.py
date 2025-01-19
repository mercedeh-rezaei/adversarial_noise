import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import urllib
import json
from fgsm_attack import fgsm_attack

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


def pgd_attack(model, input_image, target_class, epsilon=0.01, num_iterations=40, step_size=0.005):
    """
    Performs PGD attack on the image to a target class.
    """

    perturbed_image = input_image.clone().detach().requires_grad_(True).to(input_image.device)
    target = torch.tensor([target_class]).to(input_image.device)

    for i in range(num_iterations):

        output = model(perturbed_image)
        loss = -F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        perturbation = step_size * perturbed_image.grad.sign()
        perturbed_image = perturbed_image + perturbation
        # Project perturbation to epsilon-ball and clamp to [0, 1]
        perturbation = torch.clamp(perturbed_image - input_image, -epsilon, epsilon)
        perturbed_image = torch.clamp(input_image + perturbation, 0, 1)
        # torch.nn.utils.clip_grad_norm_(perturbed_image, max_norm=1.0)
        perturbed_image = perturbed_image.detach().requires_grad_(True)
    
        print(f"Max perturbation: {torch.max(perturbed_image - input_image).item()}")
        print(f"Min perturbation: {torch.min(perturbed_image - input_image).item()}")

        # Log progress
        target_confidence = output[0, target_class].item()
        print(f"Iteration {i + 1}, Target class confidence: {target_confidence}")


        imagenet_classes = load_imagenet_classes()
        if i % 10 == 0: 
            top5_indices = torch.topk(output, 5).indices.squeeze(0).cpu().numpy()
            top5_classes = [imagenet_classes[idx] for idx in top5_indices]
            print(f"Top 5 predictions: {top5_classes}")

    return perturbed_image
