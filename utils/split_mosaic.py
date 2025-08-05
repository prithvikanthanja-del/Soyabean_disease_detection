import os
import random
import shutil

source_folder = "C:/Soyabean/Soyabean_Mosaic"
train_folder = "C:/Soyabean/dataset/train/Mosaic"
val_folder = "C:/Soyabean/dataset/val/Mosaic"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

all_images = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.shuffle(all_images)

split_index = int(0.8 * len(all_images))
train_images = all_images[:split_index]
val_images = all_images[split_index:]

for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

for img in val_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print(f"✅ Mosaic: {len(train_images)} images moved to TRAIN, {len(val_images)} to VAL")
