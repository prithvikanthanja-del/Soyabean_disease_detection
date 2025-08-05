import os
import random
import shutil

# Map your real folder names to cleaned class labels
class_folders = {
    "Caterpillar and Semilooper Pest Attack": "Caterpillar",
    "Soyabean_Frog_Leaf_Eye": "Frogeye Leaf Spot",
    "Soyabean_Rust": "Rust"
}

base_input_path = "C:/Soyabean"
train_base = "C:/Soyabean/dataset/train"
val_base = "C:/Soyabean/dataset/val"

for folder_name, class_name in class_folders.items():
    source_folder = os.path.join(base_input_path, folder_name)
    train_folder = os.path.join(train_base, class_name)
    val_folder = os.path.join(val_base, class_name)

    # Create target folders
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # List and shuffle images
    all_images = [f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(all_images)

    # 80/20 split
    split_index = int(0.8 * len(all_images))
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

    for img in val_images:
        shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

    print(f"✅ {class_name}: {len(train_images)} images moved to TRAIN, {len(val_images)} to VAL")
