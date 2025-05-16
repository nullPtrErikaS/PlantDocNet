import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
SOURCE_DIR = 'dataset'         # where all class folders are now
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/val'
SPLIT_RATIO = 0.8              # 80% train / 20% val

# Ensure output folders exist
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Iterate over each class (folder)
for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    # Only use files, skip directories or hidden files
    images = [
        f for f in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, f)) and not f.startswith('.')
    ]

    if len(images) < 2:
        print(f"Skipping class '{class_name}' (only {len(images)} image)")
        continue

    train_imgs, val_imgs = train_test_split(images, train_size=SPLIT_RATIO, random_state=42)

    # Make train and val class folders
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in val_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

print("Dataset split complete.")
