# Adversarial Noise Project

This project impelements an PGD adversarial attack on the Resnet18 pre-trained model to misclassify images by adding noise.

# Running the code
To generate an adversarial image, run the following command:

The target class is an integer chosen between 1-1000 from the imagenet classes. 
The image_path is provided by the user, if not provided the dog example is used.
The epsilon has a default value of 0.03. 
The step size has a default value of 0.002.
The number of iterations has a default value of 100.

python src/main.py <target_class> <image_path> --epsilon <perturbation_strength> --num_iterations <num_iterations> --step_size <step_size>

