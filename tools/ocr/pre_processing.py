import os
import json
import pickle
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import warnings
warnings.filterwarnings("ignore")

def build_transform(input_size):
    """Build image transform for InternVL2"""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess_internvl(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamic preprocessing for InternVL2"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def preprocess_dataset(data_dir, output_dir, max_size=1344, vlm_model="internvl2-2b"):
    """
    Preprocess all images in the dataset and save preprocessed data
    """
    print(f"Starting preprocessing for {vlm_model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all image paths
    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))
    
    print(f"Found {len(all_images)} images to preprocess")
    
    # Create image metadata
    image_metadata = []
    preprocessed_data = {}
    
    for idx, img_path in enumerate(tqdm(all_images, desc="Preprocessing images")):
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            original_size = image.size
            
            # Resize if needed
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            
            # Store metadata
            filename = os.path.basename(img_path)
            metadata = {
                "index": idx,
                "filename": filename,
                "full_path": img_path,
                "original_size": original_size,
                "processed_size": image.size
            }
            image_metadata.append(metadata)
            
            # Save preprocessed image
            if vlm_model == "internvl2-2b":
                # For InternVL2, we need to do dynamic preprocessing
                transform = build_transform(input_size=448)
                images = dynamic_preprocess_internvl(image, image_size=448, use_thumbnail=True, max_num=6)
                pixel_values = [transform(img) for img in images]
                pixel_values = torch.stack(pixel_values)
                preprocessed_data[filename] = pixel_values
            else:
                # For other models, just save the resized image
                preprocessed_data[filename] = image.copy()
            
        except Exception as e:
            print(f"Error preprocessing {img_path}: {str(e)}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "image_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(image_metadata, f, indent=2)
    
    # Save preprocessed data
    preprocessed_path = os.path.join(output_dir, "preprocessed_data.pkl")
    with open(preprocessed_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"Preprocessing complete!")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Preprocessed data saved to: {preprocessed_path}")
    print(f"Total images processed: {len(preprocessed_data)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess images for VLM extraction")
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_dir", required=True, help="Output directory for preprocessed data")
    parser.add_argument("--max_size", type=int, default=1344, help="Maximum image dimension")
    parser.add_argument("--vlm_model", default="qwen2-vl", 
                        choices=["internvl2-2b", "qwen2-vl", "minicpm-v"],
                        help="VLM model to use")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_size=args.max_size,
        vlm_model=args.vlm_model
    )
