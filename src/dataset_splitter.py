import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURATION (UPDATE THESE PATHS) ---
# 1. Where are your images currently?
SRC_IMAGES_DIR = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Cape Town Highway Dataset\images"  
# 2. Where are your labels currently?
SRC_LABELS_DIR = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Cape Town Highway Dataset\labels" 
# 3. Where do you want the final ready-to-upload folder?
OUTPUT_DIR = r"C:\Users\ireoy\OneDrive - University of Cape Town\UCT ACADEMIC YEARS\UCT Masters Second Year\Masters_local_repo\Cape Town Highway Dataset\SA_Traffic_Split"

# Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
# ------------------------------------------

def split_dataset():
    # 1. Verification
    if not os.path.exists(SRC_IMAGES_DIR) or not os.path.exists(SRC_LABELS_DIR):
        print("Error: Source directories not found. Check paths.")
        return

    # 2. Create Destination Structure
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

    # 3. Match Images to Labels
    image_files = [f for f in os.listdir(SRC_IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    pairs = []
    
    print("Matching images to labels...")
    for img_file in image_files:
        basename = os.path.splitext(img_file)[0]
        txt_file = basename + ".txt"
        src_txt_path = os.path.join(SRC_LABELS_DIR, txt_file)
        
        if os.path.exists(src_txt_path):
            pairs.append((img_file, txt_file))
        else:
            print(f"Warning: No label found for {img_file}. Skipping.")

    # 4. Shuffle and Split
    random.seed(42)
    random.shuffle(pairs)
    
    n_total = len(pairs)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_set = pairs[:n_train]
    val_set = pairs[n_train:n_train+n_val]
    test_set = pairs[n_train+n_val:]
    
    print(f"Total: {n_total} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    # 5. Copy Files
    def copy_set(dataset, split_name):
        print(f"Copying {split_name}...")
        for img, txt in tqdm(dataset):
            shutil.copy2(os.path.join(SRC_IMAGES_DIR, img), os.path.join(OUTPUT_DIR, 'images', split_name, img))
            shutil.copy2(os.path.join(SRC_LABELS_DIR, txt), os.path.join(OUTPUT_DIR, 'labels', split_name, txt))

    copy_set(train_set, 'train')
    copy_set(val_set, 'val')
    copy_set(test_set, 'test')
    
    print(f"\nDone! Zip the folder '{OUTPUT_DIR}' and upload it to Google Drive.")

if __name__ == "__main__":
    split_dataset()