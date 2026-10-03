import os
import torch
from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_train_utils import evaluate_dil, classwise_accuracy, evaluate_domain
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_domain_classwise(model, loader, global_classes, domain_name, device):
    acc = evaluate_dil(model, {domain_name: loader}, device)
    class_acc = classwise_accuracy(model, loader, device, len(global_classes))
    accuracy_value = acc if isinstance(acc, float) else acc[0]
    print(f"\nDomain: {domain_name} - Overall Accuracy: {accuracy_value:.4f}")
    for i, cacc in enumerate(class_acc):
        print(f"{global_classes[i]}: {cacc:.4f}")
    return accuracy_value, class_acc

def evaluation_script(
    data_root,
    ocr_tensor_dirs,
    domains,
    global_classes,
    eaml_ckpt_path,
    batch_size=16,
    finetune_mode="head_only",
    unfreeze_depth=0,
    num_workers=4,
):
    # Initialize DIL dataloader
    class_to_idx = {c: i for i, c in enumerate(global_classes)}
    dil_loader = DILDataLoader(
        data_root=data_root,
        domain_list=domains,
        batch_size=batch_size,
        img_size=(229, 229),
        num_workers=num_workers,
        ocr_tensor_dirs=ocr_tensor_dirs,
        class_to_idx=class_to_idx
    )

    test_loaders = dil_loader.get_domain_loaders('test')
    pretrained_domain = domains[0]
    incremental_domain = domains[1]
    test_loader_pretrained = test_loaders.get(pretrained_domain)
    test_loader_incremental = test_loaders.get(incremental_domain)

    old_classes = 16
    new_classes = len(global_classes)

    # Load trained model
    model = load_eaml_model_partial(
        eaml_ckpt_path,
        old_classes,
        new_classes,
        device=DEVICE,
        text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)
    model.eval()

    print("==== Starting Domain-wise Evaluation ====")
    acc_pretrained, class_acc_pretrained = evaluate_domain_classwise(
        model, test_loader_pretrained, global_classes, pretrained_domain, DEVICE)
    acc_incremental, class_acc_incremental = evaluate_domain_classwise(
        model, test_loader_incremental, global_classes, incremental_domain, DEVICE)

    print("\n==== Evaluation Complete ====")
    print(f"Pretrained Domain ({pretrained_domain}) Overall Accuracy: {acc_pretrained:.4f}")
    print(f"Incremental Domain ({incremental_domain}) Overall Accuracy: {acc_incremental:.4f}")

    return {
        "pretrained_domain": {
            "overall_accuracy": acc_pretrained,
            "class_wise_accuracy": class_acc_pretrained
        },
        "incremental_domain": {
            "overall_accuracy": acc_incremental,
            "class_wise_accuracy": class_acc_incremental
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EAML-DIL Domain-Wise Evaluation")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ocr_tensor_dirs', nargs=2, required=True, help='OCR tensor dirs for each domain')
    parser.add_argument('--domains', required=True, help='Comma-separated domain names')
    parser.add_argument('--global_classes', required=True, help='Comma-separated global classes')
    parser.add_argument('--eaml_ckpt_path', required=True, help='Path to trained EAML checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--finetune_mode', choices=['full_finetune', 'head_only', 'partial_finetune'], default='head_only')
    parser.add_argument('--unfreeze_depth', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    domains = args.domains.split(',')
    global_classes = args.global_classes.split(',')

    results = evaluation_script(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(domains, args.ocr_tensor_dirs)},
        domains=domains,
        global_classes=global_classes,
        eaml_ckpt_path=args.eaml_ckpt_path,
        batch_size=args.batch_size,
        finetune_mode=args.finetune_mode,
        unfreeze_depth=args.unfreeze_depth,
        num_workers=args.num_workers
    )


"""
#!/bin/bash

# Paths - update these to your actual locations
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset"
OCR_TENSOR_DIR_RVL="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt"
OCR_TENSOR_DIR_TOBACCO="/home/woody/iwi5/iwi5280h/dataset/Tobacco3482_ocr_texts_tesseract.pt"
EAML_CKPT_PATH="/home/woody/iwi5/iwi5280h/dil_models/eaml_dil_ievm_training_KDILS2/best_model.pth"
#EAML_CKPT_PATH="/home/woody/iwi5/iwi5280h/emal_models/outputs/outputs/all_eaml_adamW_tesseract_20250914_221921/eaml_best_model.pt"
CHECKPOINT_DIR="/path/to/checkpoint_dir"  # If needed

# Domains and classes
DOMAINS="all_prepdataset,Tobacco3482-jpg"
GLOBAL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo,Note,Report"

# Batch size and workers
BATCH_SIZE=16
NUM_WORKERS=4
FINETUNE_MODE="head_only"
UNFREEZE_DEPTH=0

# Run evaluation
python3 utils/eaml_eval_dil.py \
  --data_dir "$DATA_DIR" \
  --ocr_tensor_dirs "$OCR_TENSOR_DIR_RVL" "$OCR_TENSOR_DIR_TOBACCO" \
  --domains "$DOMAINS" \
  --global_classes "$GLOBAL_CLASSES" \
  --eaml_ckpt_path "$EAML_CKPT_PATH" \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS \
  --finetune_mode $FINETUNE_MODE \
  --unfreeze_depth $UNFREEZE_DEPTH




"""