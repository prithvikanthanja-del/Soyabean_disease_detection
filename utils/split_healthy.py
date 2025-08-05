import os
import random
import shutil

# Your folders
source_folder = "C:/Soyabean/Healthy_Soyabean"
train_folder = "C:/Soyabean/dataset/train/Healthy"
val_folder = "C:/Soyabean/dataset/val/Healthy"

# Create destination folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get list of image files
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(all_images)

# Split into 80% train, 20% val
split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Copy images
for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

for img in val_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print(f"✅ Done! {len(train_images)} images moved to TRAIN and {len(val_images)} to VAL.")
