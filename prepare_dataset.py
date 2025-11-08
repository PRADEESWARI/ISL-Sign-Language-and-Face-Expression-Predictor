import os
import shutil
import random

# CHANGE THIS PATH to where your dataset is stored
SOURCE_DIR = "D:/EE - sem_v/Dataset/ISL Images"
DEST_DIR = "D:/EE - sem_v/isl_dataset"

TRAIN_SPLIT = 0.7  # 70% train
VAL_SPLIT = 0.2    # 20% validation
TEST_SPLIT = 0.1   # 10% test

def safe_copy(src, dest_dir, prefix):
    """Copy file with prefix to avoid overwrite."""
    filename = os.path.basename(src)
    new_filename = f"{prefix}_{filename}"
    shutil.copy(src, os.path.join(dest_dir, new_filename))

def create_dir(path):
    os.makedirs(path, exist_ok=True)

# Prepare output folders
for folder in ["train", "val", "test"]:
    create_dir(os.path.join(DEST_DIR, folder))

all_classes = set()
class_to_images = {}

# Step 1: Collect all images for each class
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            class_name = os.path.basename(root)
            if class_name in ["English Alphabet", "Numerals"]:
                continue  # skip category folders
            
            # prefix based on parent folders (kids_full, teenagers_half, etc.)
            parts = root.replace(SOURCE_DIR, "").strip("\\/").split("\\")
            prefix = "_".join([p.replace(" ", "").lower() for p in parts[:-1]])  # skip class folder
            
            img_path = os.path.join(root, file)
            class_to_images.setdefault(class_name, []).append((img_path, prefix))
            all_classes.add(class_name)

# Step 2: Shuffle, split and copy
summary = []
for class_name, img_list in class_to_images.items():
    random.shuffle(img_list)
    total = len(img_list)
    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    train_imgs = img_list[:train_end]
    val_imgs = img_list[train_end:val_end]
    test_imgs = img_list[val_end:]

    for split_name, images in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        split_dir = os.path.join(DEST_DIR, split_name, class_name)
        create_dir(split_dir)
        for img_path, prefix in images:
            safe_copy(img_path, split_dir, prefix)

    summary.append(f"{class_name}: total={total}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

# Save summary to file
with open(os.path.join(DEST_DIR, "dataset_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary))

print("✅ Dataset prepared successfully with train/val/test split!")
print(f"Classes found: {sorted(list(all_classes))}")
print(f"Summary saved at: {os.path.join(DEST_DIR, 'dataset_summary.txt')}")
