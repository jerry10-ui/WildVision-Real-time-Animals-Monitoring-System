import os
import shutil
import random

source_dir = r"C:\Coding\Projects\SEAI Project\dataset"

train_dir = os.path.join(source_dir, "train")
test_dir = os.path.join(source_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in os.listdir(source_dir):
    
    # 🔥 Skip train & test folders
    if class_name in ["train", "test"]:
        continue
    
    class_path = os.path.join(source_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue
    
    images = os.listdir(class_path)
    
    # Only take image files
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.shuffle(images)
    
    split = int(0.8 * len(images))
    
    train_images = images[:split]
    test_images = images[split:]
    
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
    
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )
    
    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

print("✅ Dataset split completed successfully!")