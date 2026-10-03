import torch
import argparse
import os

def merge_tensor_lists(input_files, output_file):
    merged = []
    for f in input_files:
        print(f"Loading {f} ...")
        data = torch.load(f, map_location="cpu")
        if not isinstance(data, list):
            raise ValueError(f"File {f} does not contain a list. Got {type(data)}.")
        merged.extend(data)
    print(f"Saving merged list with {len(merged)} entries to {output_file}")
    torch.save(merged, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple OCR .pt files (lists) into one.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .pt files to merge")
    parser.add_argument("--output", required=True, help="Output merged .pt file")
    args = parser.parse_args()
    merge_tensor_lists(args.inputs, args.output)
