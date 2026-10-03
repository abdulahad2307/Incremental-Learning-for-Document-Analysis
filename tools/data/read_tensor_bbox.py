import torch
import sys
from pathlib import Path

def read_single_ocr_tensor(dir_path, file_name):
    pt_path = Path(dir_path) / file_name
    if not pt_path.is_file():
        print(f"ERROR: File '{pt_path}' does not exist.")
        return
    data = torch.load(pt_path)
    print(f"Loaded '{pt_path}'")
    print("Keys:", list(data.keys()))
    if "input_ids" in data:
        print("input_ids shape:", data["input_ids"].shape)
    if "attention_mask" in data:
        print("attention_mask shape:", data["attention_mask"].shape)
    if "bbox" in data:
        print("bbox shape:", data["bbox"].shape)
        print("First bbox:", data["bbox"][0])
    if "bboxes" in data:
        print("bboxes shape:", data["bboxes"].shape)
        print("First polygon:", data["bboxes"][0])
    print("\nFirst 10 input_ids:", data.get("input_ids", "N/A")[:10])
    print("First 10 attention_mask:", data.get("attention_mask", "N/A")[:10])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python read_tensor_bbox.py <directory> <filename>")
        print("Example: python read_tensor_bbox.py /path/to/dir image0001.pt")
        exit(1)
    directory = sys.argv[1]
    filename = sys.argv[2]
    read_single_ocr_tensor(directory, filename)
