import torch
import torch.nn.functional as F

def normalise(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalise input images."""
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(x.device)
    return (x - mean) / std

def denormalise(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalise input images."""
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(x.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(x.device)
    return x * std + mean

def pgd_attack(model, input_image, target_class, epsilon=0.03, num_iterations=100, step_size=0.002):
    """
    Performs targeted PGD attack on the image with proper normalisation handling.
    """
    # Ensure input_image requires gradients
    perturbed_image = input_image.clone().detach().requires_grad_(True)
    original_image = input_image.clone().detach()
    
    # Convert target_class to tensor
    target = torch.tensor([target_class], device=input_image.device)
    
    # ImageNet normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    for i in range(num_iterations):
        # Forward pass
        output = model(perturbed_image)
        
        # Calculate loss for targeted attack
        target_logits = output[0, target_class]
        other_logits = torch.cat([output[0, :target_class], output[0, (target_class+1):]])
        loss = -(target_logits - torch.max(other_logits))
        
        # Backward pass
        model.zero_grad()
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        loss.backward()
        
        with torch.no_grad():
            # Convert to [0,1] space for perturbation
            denorm_original = denormalise(original_image)
            denorm_perturbed = denormalise(perturbed_image)
            
            # Calculate gradient in [0,1] space
            grad = perturbed_image.grad.data
            grad_denorm = grad.clone()
            for c in range(3):
                grad_denorm[:, c, :, :] = grad[:, c, :, :] * std[c]
            
            # Normalise gradients
            grad_norm = torch.norm(grad_denorm, p=float('inf'))
            normalised_grad = grad_denorm / (grad_norm + 1e-8)
            
            # Update image in [0,1] space
            denorm_perturbed = denorm_perturbed - step_size * normalised_grad.sign()
            
            # Project perturbation onto epsilon L-infinity ball
            perturbation = denorm_perturbed - denorm_original
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            denorm_perturbed = torch.clamp(denorm_original + perturbation, 0, 1)
            
            # Convert back to normalised space
            perturbed_image = normalise(denorm_perturbed).requires_grad_(True)
        
        # Log progress
        if i % 5 == 0:
            with torch.no_grad():
                current_output = model(perturbed_image)
                probs = F.softmax(current_output, dim=1)
                target_prob = probs[0, target_class].item()
                
                # Calculate perturbation magnitude in [0,1] space
                max_perturb = torch.max(torch.abs(denorm_perturbed - denorm_original)).item()
                
                print(f"Iteration {i + 1}/{num_iterations}")
                print(f"Target class probability: {target_prob:.4f}")
                print(f"Max perturbation: {max_perturb:.4f}")
                
                # Print top predictions every 10 iterations
                if i % 10 == 0:
                    top5_prob, top5_idx = torch.topk(probs[0], 5)
                    for prob, idx in zip(top5_prob, top5_idx):
                        print(f"Class {idx.item()}: {prob.item():.4f}")
                    print("-" * 50)
    
    return perturbed_image.detach()