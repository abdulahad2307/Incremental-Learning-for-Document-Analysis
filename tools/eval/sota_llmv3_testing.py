import os
import torch
from utils.llmv3.llmv3_data_loader import get_dataloaders
from utils.llmv3.llmv3_model_loader import LayoutLMv3
from utils.llmv3.llmv3_eval_utils import evaluate
from utils.llmv3.llmv3_train_utils import val_epoch

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LayoutLMv3 Document Classification Model")
    parser.add_argument("--dataset", type=str, choices=["rvl_cdip", "tobacco3482", "docbank", "publaynet"], required=True)
    parser.add_argument("--ocr_tensor_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--bbox_style", type=str, choices=["rect", "poly"], default="poly")
    parser.add_argument("--text_encoder", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_encoder", type=str, default="vit_base_patch16_224")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    base_classes = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "invoice", "presentation",
        "questionnaire", "resume", "memo"
    ]
    # Dataset loaders for validation/testing
    test_image_dir = os.path.join(args.image_dir, "test")
    test_loader, _, num_classes = get_dataloaders(
        dataset_name=args.dataset,
        ocr_tensor_file=args.ocr_tensor_file,
        base_classes=base_classes,
        image_dir=test_image_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        bbox_style=args.bbox_style,
        images_per_class=None,
        seed=42
    )

    # Load model and weights
    model = LayoutLMv3(
        text_model_name=args.text_encoder,
        vision_model_name=args.vision_encoder,
        num_labels=num_classes
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Evaluate model on validation/test set
    test_loss, test_acc = val_epoch(model, test_loader, device)
    print(f"Evaluation Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()


#python utils/sota_llmv3_testing.py --dataset rvl_cdip --ocr_tensor_file /home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt --image_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset --checkpoint /home/woody/iwi5/iwi5280h/FixedImage_llmv3/all_class_125k__LLMV3_AdamW_tesseract_20251111_234459/layoutlmv3_rvl_cdip_best.pt

##11
#python utils/sota_llmv3_testing.py --dataset rvl_cdip --ocr_tensor_file /home/woody/iwi5/iwi5280h/dataset/all_predataset_combined_ocr_texts_rectbbox_tesseract.pt --image_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset --checkpoint /home/woody/iwi5/iwi5280h/FixedImage_llmv3/11_class_125k_LLMV3_AdamW_tesseract_20251109_153906/layoutlmv3_rvl_cdip_best.pt
