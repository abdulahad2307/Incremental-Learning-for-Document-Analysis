import os
import torch
import torch.nn as nn
from typing import List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score
import shutil
import gc

from transformers import LayoutLMv3ForSequenceClassification

from utils.llmv3.llmv3_model_loader import LayoutLMv3
from utils.class_IL.dataloader_utils_llmv3 import get_class_il_loader, ExemplarDatasetWrapper
from utils.class_IL.train_utils_llmv3 import save_checkpoint, train_one_epoch_cil_v2, evaluate, CILMetrics
from utils.class_IL.cil_utils_llmv3 import EWC, ExemplarManager, AdaptiveLR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def expand_classifier_head(model, new_num_classes):
    old_classifier = model.classifier
    old_num_classes = old_classifier.out_features

    if new_num_classes <= old_num_classes:
        return model
    new_classifier = nn.Linear(old_classifier.in_features, new_num_classes)
    with torch.no_grad():
        new_classifier.weight[:old_num_classes] = old_classifier.weight
        if old_classifier.bias is not None:
            new_classifier.bias[:old_num_classes] = old_classifier.bias
    model.classifier = new_classifier
    return model


def run_incremental_learning(
    data_root: str,
    ocr_tensor_path: str,
    all_classes: List[str],
    base_classes: List[str],
    unseen_classes: List[str],
    base_model_path: str,
    checkpoint_dir: str,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    use_ewc: bool = False,
    lambda_ewc: float = 5000.0,
    use_exemplars: bool = True,
    max_exemplars: int = 320,
    exemplar_selection: str = "herding",
    training_mode: str = "last_layer",
    trainable_layers: Optional[List[str]] = None,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    patience: int = 10,
    max_length: int = 512,
    bbox_style: str = "rect",
    full_model_acc: Optional[float] = None,
):
    print("Starting run_incremental_learning")
    os.makedirs(checkpoint_dir, exist_ok=True)

    exemplar_mgr = ExemplarManager(
        max_exemplars=max_exemplars,
        selection_strategy=exemplar_selection
    ) if use_exemplars else None

    if unseen_classes is None or len(unseen_classes) == 0:
        raise ValueError("You must provide a non-empty list of unseen_classes")
    for cls in unseen_classes:
        if cls not in all_classes:
            raise ValueError(f"Unseen class '{cls}' is not in all_classes list")
        if cls in base_classes:
            raise ValueError(f"Unseen class '{cls}' is already in base_classes")

    start_unseen_index = 0
    start_epoch = 0
    epochs_no_improve = 0
    step_best_acc = 0.0
    step_best_path = None

    print("Model Loading Start:")
    model = LayoutLMv3(
        text_model_name="bert-base-uncased",
        vision_model_name="vit_base_patch16_224",
        num_labels=len(base_classes)
    ).to(DEVICE)
    print("Model Loading Ends")

    if resume and resume_checkpoint and os.path.exists(resume_checkpoint):
        print("Loading checkpoint for resume:", resume_checkpoint)
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE)
        print("Checkpoint loaded.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = dict(state_dict)
        if 'classifier.weight' in state_dict:
            del state_dict['classifier.weight']
        if 'classifier.bias' in state_dict:
            del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
        start_unseen_index = checkpoint.get('unseen_index', 0)
        start_epoch = checkpoint.get('epoch', 0)
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        step_best_acc = checkpoint.get('step_best_acc', 0.0)
        step_best_path = checkpoint.get('step_best_path', None)
        current_classes = checkpoint.get('current_classes', base_classes.copy())
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Resuming from unseen_index: {start_unseen_index}, epoch: {start_epoch}")
        print("Finished loading checkpoint.")
    elif os.path.exists(base_model_path):
        print("Loading base model:", base_model_path)
        checkpoint = torch.load(base_model_path, map_location=DEVICE)
        print("Base model checkpoint loaded.")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = dict(state_dict)
        if 'classifier.weight' in state_dict:
            del state_dict['classifier.weight']
        if 'classifier.bias' in state_dict:
            del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)
        current_classes = base_classes.copy()
        del checkpoint
        torch.cuda.empty_cache()
        gc.collect()
        print("Finished loading base model.")
    else:
        current_classes = base_classes.copy()
        print("No checkpoint or base model loaded, starting fresh.")

    for unseen_idx, new_class in enumerate(unseen_classes[start_unseen_index:], start=start_unseen_index):
        if unseen_idx > start_unseen_index:
            start_epoch = 0
            epochs_no_improve = 0
            step_best_acc = 0.0
            step_best_path = None

        if new_class not in current_classes:
            current_classes.append(new_class)
        print(f"\n= Incremental Step ({unseen_idx + 1}/{len(unseen_classes)}) - Adding class: {new_class} =")

        model = expand_classifier_head(model, len(current_classes))
        model = model.to(DEVICE)
        print(f"Expanded model classifier to {len(current_classes)} classes.")

        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("Frozen all model parameters except classifier.")

        exemplars_data = exemplar_mgr.get_exemplar_dataset() if (use_exemplars and exemplar_mgr is not None and unseen_idx > 0) else None

        print("Loading data loaders...")
        train_loader = get_class_il_loader(
            model_type="layoutlmv3",
            data_dir=os.path.join(data_root, "train"),
            current_classes=current_classes,
            batch_size=batch_size,
            ocr_data=ocr_tensor_path,
            max_length=max_length,
            bbox_style=bbox_style,
            exemplar_dataset=exemplars_data
        )
        val_loader = get_class_il_loader(
            model_type="layoutlmv3",
            data_dir=os.path.join(data_root, "val"),
            current_classes=current_classes,
            batch_size=batch_size,
            ocr_data=ocr_tensor_path,
            max_length=max_length,
            bbox_style=bbox_style
        )
        test_loader = get_class_il_loader(
            model_type="layoutlmv3",
            data_dir=os.path.join(data_root, "test"),
            current_classes=current_classes,
            batch_size=batch_size,
            ocr_data=ocr_tensor_path,
            max_length=max_length,
            bbox_style=bbox_style
        )
        print("Data loaders loaded.")
        torch.cuda.empty_cache()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        lr_sched = AdaptiveLR(optimizer, base_lr=lr)
        ewc = None

        this_start_epoch = start_epoch if unseen_idx == start_unseen_index else 0
        this_epochs_no_improve = epochs_no_improve if unseen_idx == start_unseen_index else 0
        this_step_best_acc = step_best_acc if unseen_idx == start_unseen_index else 0.0
        this_step_best_path = step_best_path if unseen_idx == start_unseen_index else None

        for epoch in range(this_start_epoch, num_epochs):
            print(f"\n= Epoch {epoch + 1}/{num_epochs} for Class {new_class} =")
            train_metrics = CILMetrics(current_classes)
            print("Training epoch start...")
            train_result = train_one_epoch_cil_v2(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE,
                train_metrics,
                inc_strategy=None,
                old_model=None,
                ewc=ewc if use_ewc else None,
                incremental=True
            )
            print("Training epoch done.")
            val_metrics = CILMetrics(current_classes)
            print("Validation start...")
            val_result = evaluate(model, val_loader, DEVICE, val_metrics, None)
            print("Validation done.")
            print(f"Train Loss: {train_result['loss']:.4f} | Train Acc: {train_result['top1_acc']:.4f}")
            print(f"Val Loss: {val_result['loss']:.4f} | Val Acc: {val_result['top1_acc']:.4f}")

            last_epoch_path = os.path.join(checkpoint_dir, f"epoch{epoch + 1}_{new_class}.pth")
            best_model_path = os.path.join(checkpoint_dir, f"best_model_{new_class}.pth")

            if val_result['top1_acc'] > this_step_best_acc:
                this_step_best_acc = val_result['top1_acc']
                this_epochs_no_improve = 0
                this_step_best_path = best_model_path
                save_checkpoint(model, optimizer, epoch + 1, best_model_path,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "unseen_class": new_class,
                        "epoch": epoch + 1,
                        "epochs_no_improve": this_epochs_no_improve,
                        "step_best_acc": this_step_best_acc,
                        "step_best_path": this_step_best_path
                    }
                )
                test_metrics = CILMetrics(current_classes)
                print("Evaluating test set...")
                test_result = evaluate(model, test_loader, DEVICE, test_metrics, full_model_acc)
                print("Test evaluation done.")
                print("Class-wise Test Accuracy:")
                for cls, acc in zip(current_classes, test_result['class_acc']):
                    print(f"  {cls}: {acc:.4f}")
            else:
                this_epochs_no_improve += 1
                print(f"Patience counter: {this_epochs_no_improve}/{patience}")
                save_checkpoint(model, optimizer, epoch + 1, last_epoch_path,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "unseen_class": new_class,
                        "epoch": epoch + 1,
                        "epochs_no_improve": this_epochs_no_improve,
                        "step_best_acc": this_step_best_acc,
                        "step_best_path": this_step_best_path
                    }
                )
                if this_epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1} due to no improvement in val accuracy for {patience} epochs.")
                    break

            lr_sched.step(val_result['top1_acc'])
            torch.cuda.empty_cache()
            gc.collect()
            print("Memory cleanup done after epoch.")

        if use_ewc:
            print(f"Updating EWC Fisher information for increment {unseen_idx + 1}...")
            ewc = EWC(model, train_loader, DEVICE, lambda_ewc)

        if use_exemplars and exemplar_mgr is not None:
            exemplar_mgr.update(train_loader.dataset, new_class, model)
            for cls in exemplar_mgr.exemplars:
                print(f"Exemplar samples for {cls}: {len(exemplar_mgr.exemplars[cls])}")

        if hasattr(model, "classifier"):
            print("Applying bias correction to last classifier.")
            with torch.no_grad():
                mean_bias = model.classifier.bias.mean().item()
                model.classifier.bias[:] -= mean_bias

        if this_step_best_path and os.path.exists(this_step_best_path):
            base_model_path = this_step_best_path
        else:
            print("Warning: No best checkpoint found for this increment. Skipping base_model_path update.")

        # Cleanup large objects at end of incremental step
        del train_loader, val_loader, test_loader
        del train_metrics, val_metrics, train_result, val_result, test_metrics, test_result
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleanup done after incremental step {unseen_idx + 1}.")

    final_global_best_path = None
    if this_step_best_path and os.path.exists(this_step_best_path):
        final_global_best_path = os.path.join(checkpoint_dir, "final_global_best_model.pth")
        shutil.copy(this_step_best_path, final_global_best_path)
        print(f"Final Global Best Model saved: {final_global_best_path}")
        torch.cuda.empty_cache()
    else:
        print("Warning: No best checkpoint found for the last incremental step. Not saving final global best model.")

    if final_global_best_path:
        print("\n= Final Evaluation: GLOBAL BEST MODEL on Full Datasets =")
        final_model = LayoutLMv3(
            text_model_name="bert-base-uncased",
            vision_model_name="vit_small_patch16_224",
            num_labels=len(all_classes)
        ).to(DEVICE)
        checkpoint = torch.load(final_global_best_path, map_location=DEVICE)
        final_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        final_model.eval()

        train_loader_full = get_class_il_loader(
            "layoutlmv3", os.path.join(data_root, "train"),
            all_classes, batch_size, ocr_data=ocr_tensor_path,
            max_length=max_length, bbox_style=bbox_style
        )
        val_loader_full = get_class_il_loader(
            "layoutlmv3", os.path.join(data_root, "val"),
            all_classes, batch_size, ocr_data=ocr_tensor_path,
            max_length=max_length, bbox_style=bbox_style
        )
        test_loader_full = get_class_il_loader(
            "layoutlmv3", os.path.join(data_root, "test"),
            all_classes, batch_size, ocr_data=ocr_tensor_path,
            max_length=max_length, bbox_style=bbox_style
        )

        def evaluate_with_metrics(split_name, loader):
            eval_metrics_eval = CILMetrics(class_names=all_classes.copy())
            results = evaluate(final_model, loader, DEVICE, eval_metrics_eval, full_model_acc)
            loss, acc = results.get('loss', 0), results.get('top1_acc', 0)
            preds, labels = results.get('preds', []), results.get('labels', [])
            p = precision_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            r = recall_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            f1 = f1_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            gil = (acc - full_model_acc) / (1 - full_model_acc) if full_model_acc is not None else None
            print(f"{split_name} - Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
            if gil is not None:
                print(f"{split_name} - Incremental Learning Gap (G_IL): {gil:.4f}")
            if split_name == "Test":
                class_wise_acc = results.get('class_acc', None)
                if class_wise_acc is not None:
                    print("\nClass-wise Test Accuracy:")
                    for clss, cacc in zip(all_classes, class_wise_acc):
                        print(f"  {clss}: {cacc:.4f}")
            torch.cuda.empty_cache()

        evaluate_with_metrics("Train", train_loader_full)
        evaluate_with_metrics("Validation", val_loader_full)
        evaluate_with_metrics("Test", test_loader_full)

        del train_loader_full, val_loader_full, test_loader_full, final_model
        gc.collect()
        torch.cuda.empty_cache()
        print("Memory cleanup done after final evaluation.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_path', required=True)
    p.add_argument('--all_classes', required=True)
    p.add_argument('--base_classes', required=True)
    p.add_argument('--unseen_classes', required=True)
    p.add_argument('--base_model_path', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--accum_steps', type=int, default=1, help="Gradient accumulation steps")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--max_exemplars', type=int, default=320)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='herding')
    p.add_argument('--training_mode', choices=['full','last_layer','selective'], default='last_layer')
    p.add_argument('--trainable_layers', nargs='+', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume_checkpoint', type=str, default=None)
    p.add_argument('--full_model_acc', type=float, default=None, help="Full model accuracy")
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--bbox_style', type=str, default="rect")
    
    args = p.parse_args()

    run_incremental_learning(
        data_root=args.data_dir,
        ocr_tensor_path=args.ocr_tensor_path,
        all_classes=args.all_classes.split(','),
        base_classes=args.base_classes.split(','),
        unseen_classes=args.unseen_classes.split(','),
        base_model_path=args.base_model_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        use_ewc=args.use_ewc,
        lambda_ewc=args.lambda_ewc,
        use_exemplars=args.use_exemplars,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        training_mode=args.training_mode,
        trainable_layers=args.trainable_layers,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        patience=args.patience,
        max_length=args.max_length,
        bbox_style=args.bbox_style,
        full_model_acc=args.full_model_acc    
    )
