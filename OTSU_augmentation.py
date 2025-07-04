import os
from torchvision import transforms
from PIL import Image
#import random
#import torch
import numpy as np
import cv2

# seed = 42
# random.seed(seed)
# torch.manual_seed(seed)

# Parameters
input_dir = "C:/Users/bonzi/Desktop/FYP_Project/OTSU_Contrast_Crop_Latest_PNG"  # Otsu thresholded input
output_dir = "C:/Users/bonzi/Desktop/FYP_Project/thesis_augment"  # Output directory
image_size = 224
num_augmentations = 25

# Define transformations for grayscale images (no .convert("RGB"))
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),  # Resize to 224x224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20, interpolation=Image.BICUBIC),  # Random rotation within ±20 degrees
    transforms.RandomAffine(degrees=20, translate=(0.05, 0.05), scale=(0.8, 1.05), interpolation=Image.BICUBIC),  # Random affine transformation
])

os.makedirs(output_dir, exist_ok=True)

# Process each class folder
for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    if not os.path.isdir(class_input_path):
        continue

    os.makedirs(class_output_path, exist_ok=True)

    for image_name in os.listdir(class_input_path):
        image_path = os.path.join(class_input_path, image_name)

        try:
            image = Image.open(image_path).convert("L")  # Grayscale (Otsu)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue

        for i in range(num_augmentations):
            augmented_image = transform(image)

            # Convert to numpy array for pixel cleaning
            np_image = np.array(augmented_image)

            # Apply OTSU thresholding 
            _, otsu_image = cv2.threshold(np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply your idea: set all 0 < pixel < 255 to 255 (remove grey)
            np_image_cleaned = np.where((otsu_image > 0) & (otsu_image < 255), 255, otsu_image)

            # Convert back to PIL Image
            cleaned_image = Image.fromarray(np.uint8(np_image_cleaned))

            # Save the image
            augmented_image_name = f"{os.path.splitext(image_name)[0]}_aug{i + 1}.png"
            augmented_image_path = os.path.join(class_output_path, augmented_image_name)
            cleaned_image.save(augmented_image_path)

        print(f"Processed {image_name} for class {class_name}.")

print("Augmentation and cleaning completed. Cleaned images saved in:", output_dir)
