import os
import shutil
import random

# Configurations
src_dir = r"C:/Users/bonzi/Desktop/FYP_Project_TF/augmented_binary_final"  
# Original dataset folder: each class is in its own subfolder

dst_dir = r"C:/Users/bonzi/Desktop/FYP_Project_TF/binary_splitdata"  
# Destination folder where the new structure will be created

train_ratio = 0.8  # 80% for training, 20% for validation

# Create destination subfolders "train" and "validation"
train_dst = os.path.join(dst_dir, "train")
val_dst   = os.path.join(dst_dir, "validation")
os.makedirs(train_dst, exist_ok=True)
os.makedirs(val_dst, exist_ok=True)

# Process each class folder in the original dataset
for class_name in os.listdir(src_dir):
    class_src = os.path.join(src_dir, class_name)
    if not os.path.isdir(class_src):
        continue  # Skip non-folders

    # List all image files in the class folder
    images = [f for f in os.listdir(class_src) if os.path.isfile(os.path.join(class_src, f))]
    random.shuffle(images)  # Randomize the order

    # Determine split index
    split_index = int(train_ratio * len(images))
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class-specific subfolders in train and validation destination folders
    class_train_dst = os.path.join(train_dst, class_name)
    class_val_dst = os.path.join(val_dst, class_name)
    os.makedirs(class_train_dst, exist_ok=True)
    os.makedirs(class_val_dst, exist_ok=True)

    # Copy training images
    for img in train_images:
        src_path = os.path.join(class_src, img)
        dst_path = os.path.join(class_train_dst, img)
        shutil.copy(src_path, dst_path)

    # Copy validation images
    for img in val_images:
        src_path = os.path.join(class_src, img)
        dst_path = os.path.join(class_val_dst, img)
        shutil.copy(src_path, dst_path)

print("Data splitting and copying complete.")