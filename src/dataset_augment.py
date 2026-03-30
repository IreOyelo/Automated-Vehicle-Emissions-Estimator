import os
import cv2
import glob
import random
import albumentations as A
from tqdm import tqdm

# --- CONFIGURATION ---
# Path to your clean TRAINING images (already split)
TRAIN_IMG_DIR = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Cape Town Highway Dataset\SA_Traffic_Split\images\train"
TRAIN_LBL_DIR = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Cape Town Highway Dataset\SA_Traffic_Split\labels\train"

# How many augmented versions to create?
# 0.5 means we add 50% more images (e.g. 1000 images -> 1500 images)
AUGMENTATION_RATIO = 0.5 
# ---------------------

def create_augmentations():
    print(f"--- STARTING THESIS AUGMENTATION PIPELINE ---")
    
    # 1. Define the "Weather" & "CCTV" Pipeline
    # We combine them into a single OneOf block or Sequential block
    transform = A.Compose([
        # A. Weather Challenge (One of these will happen)
        A.OneOf([
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=1.0),
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.5, alpha_coef=0.08, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), # Contrast Enhance
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ], p=0.7), # Apply weather logic 70% of the time

        # B. CCTV Quality Challenge (One of these will happen)
        A.OneOf([
            A.MotionBlur(blur_limit=15, p=1.0),  # Simulates 120km/h blur
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0), # Sensor Grain
            A.ImageCompression(quality_lower=10, quality_upper=50, p=1.0), # JPEG Blocking
        ], p=0.5),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # 2. Get Files
    img_files = glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")) + \
                glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png"))
    
    # Select random subset to augment
    num_to_aug = int(len(img_files) * AUGMENTATION_RATIO)
    files_to_aug = random.sample(img_files, num_to_aug)
    
    print(f"Original Training Set: {len(img_files)} images")
    print(f"Generating {num_to_aug} new synthetic 'Hard Samples'...")

    for img_path in tqdm(files_to_aug):
        # Load Image
        image = cv2.imread(img_path)
        if image is None: continue
        h, w = image.shape[:2]
        
        # Load Label
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(TRAIN_LBL_DIR, base_name + ".txt")
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    # Normalize YOLO format
                    coords = [float(x) for x in parts[1:]] 
                    if len(coords) == 4:
                        bboxes.append(coords)
                        class_labels.append(cls)

        # Apply Transform
        try:
            if len(bboxes) > 0:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']
            else:
                # Background image (no boxes)
                transformed = transform(image=image, bboxes=[], class_labels=[])
                aug_img = transformed['image']
                aug_bboxes = []
                aug_labels = []
            
            # Save Augmented Version
            # We prefix with 'aug_' so we know which are synthetic
            new_name = f"aug_{base_name}"
            
            # Save Image
            cv2.imwrite(os.path.join(TRAIN_IMG_DIR, new_name + ".jpg"), aug_img)
            
            # Save Label
            with open(os.path.join(TRAIN_LBL_DIR, new_name + ".txt"), 'w') as f:
                for cls, box in zip(aug_labels, aug_bboxes):
                    # Ensure normalized 0-1
                    box = [max(0.0, min(1.0, x)) for x in box]
                    line = f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
                    f.write(line)
                    
        except Exception as e:
            print(f"Skip {base_name}: {e}")

    print("Success. Dataset Expanded.")

# Run it
create_augmentations()