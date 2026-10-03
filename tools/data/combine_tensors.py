import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--key', required=True, help='Key to extract from each dict, e.g. input_ids')
    parser.add_argument('--dim', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensors = []
    for file in args.inputs:
        obj = torch.load(file, map_location=device)
        print(f"{file} contains keys: {list(obj.keys())}")
        tensor = obj[args.key]
        tensors.append(tensor)
    combined = torch.cat(tensors, dim=args.dim)
    torch.save(combined, args.output)
    print(f"Merged {len(args.inputs)} tensors under key '{args.key}' into {args.output} (shape: {tuple(combined.shape)})")

if __name__ == '__main__':
    main()
