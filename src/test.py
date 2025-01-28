import torch
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import torchvision.transforms as transforms
from pgd_attack import pgd_attack
import json
import os
import random
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from utils import save_image, denorm

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

def setup_model_and_transforms(device):
    """Setup model and transforms."""
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return model, preprocess

def evaluate_attack(model, image_tensor, target_class, imagenet_classes):
    """Evaluate model predictions before and after attack."""
    with torch.no_grad():
        # Original prediction
        orig_output = model(image_tensor)
        orig_prob, orig_class = torch.max(torch.softmax(orig_output, dim=1), 1)
        orig_class = orig_class.item()
        orig_prob = orig_prob.item()
    
    # Perform attack
    perturbed_image = pgd_attack(
        model=model,
        input_image=image_tensor.clone(),  # Clone to ensure we don't modify original
        target_class=target_class,
        epsilon=0.03,
        num_iterations=100,
        step_size=0.002
    )
    
    with torch.no_grad():
        # Evaluate attack result
        final_output = model(perturbed_image)
        final_probs = torch.softmax(final_output, dim=1)
        final_prob, final_class = torch.max(final_probs, 1)
        final_class = final_class.item()
        final_prob = final_prob.item()
        target_prob = final_probs[0, target_class].item()
        
        # Calculate perturbation magnitude
        max_perturbation = torch.max(torch.abs(perturbed_image - image_tensor)).item()
        
        return {
            'original_class': orig_class,
            'original_class_name': imagenet_classes.get(orig_class, 'Unknown'),
            'original_probability': orig_prob,
            'target_class': target_class,
            'target_class_name': imagenet_classes.get(target_class, 'Unknown'),
            'final_class': final_class,
            'final_class_name': imagenet_classes.get(final_class, 'Unknown'),
            'target_probability': target_prob,
            'final_probability': final_prob,
            'max_perturbation': max_perturbation,
            'success': final_class == target_class,
            'perturbed_image': perturbed_image
        }

def run_test_suite(image_path, num_classes=50, seed=42, output_dir='test_results'):
    """Run comprehensive test suite on adversarial attack."""
    # Set random seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    
    # Load model and setup transforms
    model, preprocess = setup_model_and_transforms(device)
    
    # Load ImageNet classes
    imagenet_classes = load_imagenet_classes()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Select target classes
    num_total_classes = 1000
    target_classes = random.sample(range(num_total_classes), num_classes)
    
    # Store results
    results = []
    
    # Run tests
    for i, target_class in enumerate(target_classes):
        print(f"\nTesting target class {target_class} ({i+1}/{num_classes})")
        print(f"Target class name: {imagenet_classes.get(target_class, 'Unknown')}")
        
        try:
            # Evaluate attack
            result = evaluate_attack(model, image_tensor, target_class, imagenet_classes)
            results.append(result)
            
            # Save perturbed image
            if result['success']:
                save_image(
                    result['perturbed_image'],
                    f"{output_dir}/images/success_target_{target_class}.jpg"
                )
        except Exception as e:
            print(f"Error testing class {target_class}: {str(e)}")
            continue
    
    # Create results DataFrame
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'perturbed_image'}
        for r in results
    ])
    
    # Save results
    df.to_csv(f"{output_dir}/results.csv", index=False)
    
    # Calculate and save statistics
    stats = {
        'total_tests': len(results),
        'successful_attacks': sum(r['success'] for r in results),
        'success_rate': sum(r['success'] for r in results) / len(results) if results else 0,
        'avg_perturbation': sum(r['max_perturbation'] for r in results) / len(results) if results else 0,
        'avg_target_prob': sum(r['target_probability'] for r in results) / len(results) if results else 0
    }
    
    with open(f"{output_dir}/statistics.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Create visualization
    if results:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [r['max_perturbation'] for r in results],
            [r['target_probability'] for r in results],
            c=['g' if r['success'] else 'r' for r in results],
            alpha=0.6
        )
        plt.xlabel('Maximum Perturbation')
        plt.ylabel('Target Class Probability')
        plt.title('Attack Performance')
        plt.savefig(f"{output_dir}/performance_plot.png")
        plt.close()
    
    print("\nTest Results Summary:")
    print(f"Total tests: {stats['total_tests']}")
    print(f"Successful attacks: {stats['successful_attacks']}")
    print(f"Success rate: {stats['success_rate']*100:.2f}%")
    print(f"Average perturbation: {stats['avg_perturbation']:.4f}")
    print(f"Average target probability: {stats['avg_target_prob']:.4f}")
    print(f"\nDetailed results saved to: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test adversarial attack across multiple classes")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--num_classes", type=int, default=50,
                        help="Number of target classes to test (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Directory to save results (default: test_results)")
    
    args = parser.parse_args()
    run_test_suite(
        args.image_path,
        num_classes=args.num_classes,
        seed=args.seed,
        output_dir=args.output_dir
    )