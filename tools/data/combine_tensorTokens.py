import torch
import argparse
import os

def merge_tensor_dicts(input_files, output_file, overwrite=False):
    merged = {}
    for f in input_files:
        print(f"Loading {f} ...")
        data = torch.load(f, map_location="cpu")
        if not isinstance(data, dict):
            raise ValueError(f"File {f} does not contain a dict. Got {type(data)}.")
        for k in data:
            if k in merged:
                print(f"Warning: Duplicate key '{k}' found in {f}. Overwriting previous entry.")
                if not overwrite:
                    continue
            merged[k] = data[k]
    print(f"Saving merged dict with {len(merged)} entries to {output_file}")
    torch.save(merged, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple OCR tensor (.pt) files (dicts) into one.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input .pt files to merge")
    parser.add_argument("--output", required=True, help="Output merged .pt file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite duplicate keys (default: keep first)")
    args = parser.parse_args()
    merge_tensor_dicts(args.inputs, args.output, overwrite=args.overwrite)
