# this script applies OTSU and contrast stretching
import os
import cv2
import numpy as np

# Paths
input_root = "C:/Users/bonzi/Desktop/FYP_Project/thesis_use"
output_root = "C:/Users/bonzi/Desktop/FYP_Project/result_thesis"

# Create output root folder if not exists
os.makedirs(output_root, exist_ok=True)

# Loop over class folders
for class_name in os.listdir(input_root):
    input_class_path = os.path.join(input_root, class_name)
    output_class_path = os.path.join(output_root, class_name)

    # Skip if not a directory
    if not os.path.isdir(input_class_path):
        print(f"Skipping non-directory: {input_class_path}")
        continue

    # Create corresponding output class folder
    os.makedirs(output_class_path, exist_ok=True)

    # Loop over each image in the class folder
    for img_name in os.listdir(input_class_path):
        input_img_path = os.path.join(input_class_path, img_name)
        output_img_path = os.path.join(output_class_path, img_name)

        # Read image
        img = cv2.imread(input_img_path)
        if img is None:
            print(f"Failed to read image: {input_img_path}")
            continue

        # Grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply contrast stretching
        min_val = np.min(gray)
        max_val = np.max(gray)

        if max_val - min_val == 0:
            stretched = gray.copy()  # Avoid division by zero
        else:
            stretched = ((gray - min_val) / (max_val - min_val)) * 255
            stretched = stretched.astype(np.uint8)

        # Gaussian blur
        # blurred = cv2.GaussianBlur(stretched, (5, 5), 0)

        # Median filter
        blurred = cv2.medianBlur(stretched, 5)

        # Otsu's thresholding
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # # Morphological operations
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)  # fill black holes/gaps inside hand region
        # opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1) # remove white noise outside hand region

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=4)  # fill black holes/gaps inside hand region
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=8)  # remove white noise outside hand region

        # Save the result
        # base_name, ext = os.path.splitext(img_name)
        # output_name = f"{base_name}_stretch_otsu{ext}"
        # output_path = os.path.join(output_class_path, output_name)

        # save the result in png format
        base_name,_= os.path.splitext(img_name)
        output_name = f"{base_name}_morpho.png"
        output_path = os.path.join(output_class_path, output_name)

        cv2.imwrite(output_path, opened) 

        # Print progress
        print(f"Processed: {img_name} for class {class_name}")

print("All images processed with Contrast Stretching and Otsu's thresholding.")
