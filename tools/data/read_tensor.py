import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description="Read and inspect a PyTorch .pt tensor file.")
    parser.add_argument("--file", required=True, help="Path to the .pt file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to load the tensor on (default: auto-detect)")
    args = parser.parse_args()

    obj = torch.load(args.file, map_location=args.device)

    print(f"\nLoaded object from {args.file}:")
    print(f"Type: {type(obj)}")

    if isinstance(obj, torch.Tensor):
        print(f"Tensor shape: {obj.shape}")
        print(f"Tensor dtype: {obj.dtype}")
    elif isinstance(obj, dict):
        print(f"Dict keys: {list(obj.keys())}")
        for k, v in obj.items():
            if isinstance(v, torch.Tensor):
                print(f"  Key: {k} | Tensor shape: {v.shape} | dtype: {v.dtype}")
            else:
                print(f"  Key: {k} | Type: {type(v)}")
    else:
        print("Loaded object is neither a tensor nor a dict. Type:", type(obj))

if __name__ == "__main__":
    main()
