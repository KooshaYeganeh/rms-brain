import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
source_dir = "/run/media/koosha/Dataset"  # Update with your dataset root directory
train_dir = "/run/media/koosha/Dataset/train"
val_dir = "/run/media/koosha/Dataset/val"

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Iterate over each category folder
for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if not os.path.isdir(category_path):
        continue  # Skip if not a directory

    # Get list of all files in the category
    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    # Skip if the folder has no files
    if len(files) == 0:
        print(f"Skipping empty category folder: {category}")
        continue

    # Split files into train and validation sets
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    # Create corresponding directories in train and val
    train_category_dir = os.path.join(train_dir, category)
    val_category_dir = os.path.join(val_dir, category)
    os.makedirs(train_category_dir, exist_ok=True)
    os.makedirs(val_category_dir, exist_ok=True)

    # Move files to train and val directories
    for file in train_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(train_category_dir, file))
    for file in val_files:
        shutil.copy(os.path.join(category_path, file), os.path.join(val_category_dir, file))

print("Data has been split into train and val directories.")
