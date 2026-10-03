import json
import argparse
import os

def merge_ocr_results(input_files, output_file):
    """
    Merge multiple OCR JSON files into a single file
    
    Args:
        input_files: List of input JSON file paths
        output_file: Output merged JSON file path
    """
    merged_results = {}
    
    print(f"Merging {len(input_files)} JSON files...")
    
    for i, input_file in enumerate(input_files):
        print(f"Processing file {i+1}/{len(input_files)}: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"Warning: File not found, skipping: {input_file}")
            continue
            
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check for duplicate keys
        duplicates = set(merged_results.keys()) & set(data.keys())
        if duplicates:
            print(f"  Warning: Found {len(duplicates)} duplicate image paths")
        
        merged_results.update(data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    print(f"Merge completed! Total images: {len(merged_results)}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge OCR JSON results")
    parser.add_argument('--input_files', nargs='+', required=True, 
                       help='List of input JSON files')
    parser.add_argument('--output_file', required=True,
                       help='Output JSON file')
    args = parser.parse_args()
    
    merge_ocr_results(args.input_files, args.output_file)
