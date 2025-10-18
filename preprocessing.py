import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# ===== CONFIGURATION =====
RAW_DATA_DIR = 'raw_data'   # Folder containing the 10 classes
OUTPUT_DIR = 'dataset'           # Final output directory
IMG_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 2      # number of augmented copies per original image
SPLIT_RATIOS = (0.7, 0.15, 0.15) # train/val/test split

# ===== SETUP AUGMENTATION =====
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# ===== CLEAN OLD OUTPUT IF EXISTS =====
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# ===== STEP 1: PROCESS EACH CLASS =====
for class_name in os.listdir(RAW_DATA_DIR):
    class_path = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\nProcessing class: {class_name}")

    images = []
    # Read and resize all images
    for img_name in tqdm(os.listdir(class_path)):
        img_path = os.path.join(class_path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img, IMG_SIZE)
            images.append(resized)
        except Exception as e:
            print(f"Skipping corrupt file: {img_name}")

    images = np.array(images)
    if len(images) == 0:
        print(f"No valid images found in {class_name}, skipping...")
        continue

    # ===== STEP 2: SPLIT =====
    train, temp = train_test_split(images, test_size=(1 - SPLIT_RATIOS[0]), random_state=42)
    val, test = train_test_split(temp, test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), random_state=42)

    splits = {'train': train, 'val': val, 'test': test}

    # ===== STEP 3: SAVE IMAGES & AUGMENT =====
    for split_name, split_data in splits.items():
        save_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
        os.makedirs(save_dir, exist_ok=True)

        for i, img in enumerate(split_data):
            filename = f"{class_name}_{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # Data Augmentation
            x = img_to_array(img)
            x = np.expand_dims(x, 0)
            aug_iter = datagen.flow(x, batch_size=1)
            for j in range(AUGMENTATIONS_PER_IMAGE):
                aug_img = next(aug_iter)[0].astype('uint8')
                aug_filename = f"{class_name}_{i}_aug{j}.jpg"
                cv2.imwrite(os.path.join(save_dir, aug_filename), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

print("\n✅ Preprocessing Complete!")
print(f"Processed dataset saved in: {OUTPUT_DIR}")
