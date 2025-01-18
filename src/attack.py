import torch
import torch.nn.functional as F

def fgsm_attack(model, input_image, target_class, epsilon=0.01):

    """
    Performs FGSM attack on the image to a target class

    """
    model.eval()
    input_image.requires_grad=True
    output = model(input_image)

    target = torch.tensor([target_class]).to(input_image.device)
    loss = F.cross_entropy(output, target)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * input_image.grad.sign()

    print(f"Perturbation values: {perturbation.mean().item()}")

    adversarial_image = input_image + perturbation
    # adding clipping to keep the range in [0,1]
    adversarial_image = torch.clamp(adversarial_image, 0, 1)

    print(f"Original image mean: {input_image.mean().item()}, Perturbed image mean: {adversarial_image.mean().item()}")

    return adversarial_image