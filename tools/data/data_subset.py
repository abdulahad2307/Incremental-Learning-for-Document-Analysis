import os
import random
import shutil

def create_subset(data_dir,out_dir, subset_name="subset", subset_size_per_class=200, seed=42):
    """
    Creating a subset of the dataset within the original directory structure.
    
    Args:
        data_path (str): Path to original dataset (contains train/val/test folders)
        subset_name (str): Name of the subset folder to create
        subset_size_per_class (int): Number of images per class to include in subset
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    splits = ['train', 'val', 'test']
    
    subset_root = os.path.join(out_dir, subset_name)
    os.makedirs(subset_root, exist_ok=True)
    
    for split in splits:
        original_split_path = os.path.join(data_dir, split)
        subset_split_path = os.path.join(subset_root, split)
        
        if not os.path.exists(original_split_path):
            continue
            
        # Getting all class folders
        classes = [d for d in os.listdir(original_split_path) 
                  if os.path.isdir(os.path.join(original_split_path, d))]
        
        for class_name in classes:
            original_class_path = os.path.join(original_split_path, class_name)
            subset_class_path = os.path.join(subset_split_path, class_name)
            
            os.makedirs(subset_class_path, exist_ok=True)
            
            # Getting all image files
            images = [f for f in os.listdir(original_class_path) 
                     if f.endswith('.tif')]
            
            if split == 'train':
                subset_size_per_class=140
                # If there are fewer images than requested, take all
                sample_size = min(subset_size_per_class, len(images))
            elif split == 'val':
                subset_size_per_class=40
                sample_size = min(subset_size_per_class, len(images))
            elif split == 'test':
                subset_size_per_class=20
                sample_size = min(subset_size_per_class, len(images))
            # Randomly selecting images
            selected_images = random.sample(images, sample_size)
            
            # Creating symbolic links to original images (saves space)
            for img in selected_images:
                src = os.path.join(original_class_path, img)
                dst = os.path.join(subset_class_path, img)
                
                shutil.copy2(src, dst)  # creates actual copies
                # os.symlink(src, dst)  # creates symbolic links (saves space)
                
            print(f"Created links for {sample_size} images from {original_class_path} to {subset_class_path}")

data_dir = "/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
out_dir = "/home/woody/iwi5/iwi5280h/dataset/"
create_subset(data_dir, out_dir,  subset_name="small_dataset2", subset_size_per_class=200)