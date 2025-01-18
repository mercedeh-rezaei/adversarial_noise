# Adversarial Noise Project

This project impelements an FGSM adversarial attack on the Resnet18 pre-trained model to misclassify images by adding noise.

# Running the code
To generate an adversarial image, run the following command

python src/main.py <image_path> <target_class> --output_path <output_path> --epslion <perturbation_strength>
