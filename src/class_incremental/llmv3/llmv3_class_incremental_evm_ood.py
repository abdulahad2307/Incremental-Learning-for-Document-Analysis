import copy
import argparse
import os
import torch
import torch.nn.functional as F
import gc
from torch.utils.data import DataLoader, ConcatDataset
from utils.seed import add_seed_arg, set_seed
from utils.llmv3.llmv3_model_loader import load_llmv3_checkpoint
from utils.llmv3.llmv3_il_common import build_cil_train_loader, add_il_args, make_teacher, distill_term, center_classifier_bias, gil as gil_ratio, fit_evm, evm_open_set_eval
from utils import run_log
from utils.evm.evm_loss import evm_nll_loss
from utils.evm.evm_state import load_open_set_state, save_open_set_state
from utils.il_checks import warn_inactive_terms
from utils.training_scope import set_training_scope
from utils.ood.vim import VIM_OOD
from utils.ood.ood_loss import vim_ood_loss
from utils.ood.ood_eval import make_detector, evaluate_ood, collect_features_logits, last_linear_layer, subsample_loader
from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader
from utils.llmv3.llmv3_incremental_utils import (
    EWC, distillation_loss, BiasCorrectionLayer,
    ExemplarHandler, expand_classifier, evaluate
)
import torch.optim as optim
import torch.cuda.amp as amp
from tqdm import tqdm

# Added imports for EVM integration
from utils.evm.evm_classifier import EVMClassifier
from utils.evm.evm_eval import evm_openset_metrics
from utils.class_IL.cil_utils import extract_feature_vectors, extract_feature_vectors2


def parse_args():
    parser = argparse.ArgumentParser(description="LayoutLMv3 Class-Incremental Training with EVM + OOD (L_EVM + L_OOD)")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--ocr_tensor_path', required=True)
    parser.add_argument('--all_classes', required=True)
    parser.add_argument('--base_classes', required=True)
    parser.add_argument('--unseen_classes', required=True)
    parser.add_argument('--dataset_name', default='rvl_cdip', choices=['rvl_cdip', 'tobacco3482'])
    parser.add_argument('--base_model_path', required=True)
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--use_ewc', action='store_true')
    parser.add_argument('--lambda_ewc', type=float, default=5000.0)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_exemplars', type=int, default=16)
    parser.add_argument('--exemplar_selection', default='herding', choices=['random', 'herding'])
    parser.add_argument('--training_mode', default='last_layer', choices=['classifier_only', 'last_layer', 'full_model', 'full'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_checkpoint')
    parser.add_argument('--full_model_acc', type=float, default=None)
    parser.add_argument('--images_per_class', type=int, default=12500, help="Max images per unseen class")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--test_classes', default=None, help="Comma separated test classes if different")
    add_il_args(parser, strategy_default="standard", evm=True, ood=True, persist=True, cil=True)
    add_seed_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_log.init("llmv3", "CIL", "EVM+OOD", args, new_class=args.unseen_classes)
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    all_classes = [c.strip() for c in args.all_classes.split(',')]
    base_classes = [c.strip() for c in args.base_classes.split(',')]
    unseen_classes = [c.strip() for c in args.unseen_classes.split(',')]
    # Model head order: base classes (as the base model was trained) followed by the new classes
    label_space = base_classes + unseen_classes

    if args.test_classes:
        test_classes = [c.strip() for c in args.test_classes.split(',')]
    else:
        test_classes = base_classes + unseen_classes

    base_num_classes = len(base_classes)
    total_num_classes = len(all_classes)

    base_model_acc = args.base_model_acc
    print("Loading Base Model.....")
    base_model, checkpoint = load_llmv3_checkpoint(args.base_model_path, base_num_classes, device)  # custom or hf, from the checkpoint
    # Frozen teacher = the base model before its classifier is expanded (the student is trained in place)
    teacher_model = make_teacher(base_model, args.strategy)
    expand_classifier(base_model, base_num_classes, base_num_classes+1, device)
    model = base_model
    print("Loading Base Model..... Complete!")
    del checkpoint
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize EVM classifier
    evm = EVMClassifier(tailsize=args.evm_tailsize, cover_threshold=args.evm_threshold)
    ood_vim = VIM_OOD()  # ViM behind the training-time L_OOD
    if args.evm_persist:
        # Continue with the EVM / ViM of the job that produced the base model
        load_open_set_state(args.base_model_path, evm=evm, vim=ood_vim)

    set_training_scope(model, args.training_mode)


    print("Loading base train data for exemplar selection .....")
    base_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes,
        label_classes=label_space,
        image_dir=args.data_dir,
        split="train", 
        batch_size=args.batch_size,
        images_per_class=args.max_exemplars,
        seed=args.seed,
    )
    print(f"Base training samples: {len(base_train_loader.dataset)}")

    print("Loading unseen train data .....")
    unseen_train_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=unseen_classes,
        label_classes=label_space,
        image_dir=args.data_dir,
        split="train", 
        batch_size=args.batch_size,
        images_per_class=args.images_per_class,
        seed=args.seed,
    )
    print(f"Unseen class samples (limited): {len(unseen_train_loader.dataset)}")

    val_test_img = 1250
    print("Loading validation data .....")
    val_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes + unseen_classes,
        label_classes=label_space,
        image_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        images_per_class=val_test_img,
        seed=args.seed,
    )
    print(f"Validation samples: {len(val_loader.dataset)}")

    print("Loading test data .....")
    test_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name,
        ocr_tensor_file=args.ocr_tensor_path,
        classes=base_classes + unseen_classes,
        label_classes=label_space,
        image_dir=args.data_dir,
        split="test",
        batch_size=args.batch_size,
        images_per_class=val_test_img,
        seed=args.seed,
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    print("Preparing exemplars from base classes ...")
    base_class_indices = [label_space.index(c) for c in base_classes]
    exemplar_handler = ExemplarHandler(max_exemplars_per_class=args.max_exemplars, selection_method=args.exemplar_selection)
    exemplar_handler.update_exemplars(base_train_loader.dataset, base_class_indices, model, device)
    exemplar_samples = exemplar_handler.get_exemplar_dataset()

    # New class + exemplars (default), or all seen classes with --joint_training; optional balanced sampling
    combined_loader = build_cil_train_loader(args, unseen_train_loader, exemplar_samples, label_space)
    combined_dataset = combined_loader.dataset
    print("Exemplar dataset prepared.")

    print(f"Total exemplar samples: {len(exemplar_samples)}")
    print(f"Total unseen train samples: {len(unseen_train_loader.dataset)}")
    print(f"Combined training dataset samples: {len(combined_dataset)}")

    ewc = EWC(model, base_train_loader, device, fisher_n=500, lambda_ewc=args.lambda_ewc) if args.use_ewc else None

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)

    start_epoch = 0
    best_val_acc = 0.0
    patience_counter = 0

    if args.resume and args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)

    print("Training started...")

    # Initialize GradScaler for mixed precision
    scaler = amp.GradScaler()

    # Freeze all layers except last classifier layer as before
    set_training_scope(model, args.training_mode)

    warn_inactive_terms(model, use_ewc=args.use_ewc, lambda_evm=args.lambda_evm, lambda_ood=args.lambda_ood)
    for epoch in range(start_epoch, args.num_epochs):
        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        loop = tqdm(combined_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)

        for i, batch in enumerate(loop):
            for k in ['input_ids', 'attention_mask', 'bbox', 'pixel_values']:
                batch[k] = batch[k].to(device)

            batch['labels'] = torch.tensor(
                [label_space.index(lbl) if isinstance(lbl, str) else lbl for lbl in batch['labels']],
                device=device
            )

            optimizer.zero_grad()

            with amp.autocast():
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                old_logits = None
                if teacher_model is not None:
                    with torch.no_grad():
                        old_logits = teacher_model(**inputs)

                feats = model.forward_features(batch['input_ids'], batch['bbox'], batch['attention_mask'], batch['pixel_values'])
                new_logits_raw = model.classifier(feats)

                cls_loss = F.cross_entropy(new_logits_raw, batch['labels'])
                loss = cls_loss
                if old_logits is not None:
                    loss = loss + args.lambda_distill * distill_term(new_logits_raw, old_logits, args.temperature)
                evm_term = evm_nll_loss(evm, feats, batch['labels'], class_names=label_space)
                if evm_term is not None:
                    loss = loss + args.lambda_evm * evm_term
                # L_OOD: ViM virtual-logit penalty on samples of classes the ViM was fitted on
                ood_term = vim_ood_loss(ood_vim, feats, new_logits_raw, batch['labels'], class_names=label_space)
                if ood_term is not None:
                    loss = loss + args.lambda_ood * ood_term
                if ewc:
                    loss += ewc.penalty(model)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch['labels'].size(0)
            _, predicted = new_logits_raw.max(1)
            total += batch['labels'].size(0)
            correct += predicted.eq(batch['labels']).sum().item()

            # Aggressive cleanup
            del batch, old_logits, new_logits_raw, loss, cls_loss, predicted #kd_loss
            torch.cuda.empty_cache()
            gc.collect()

            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            train_acc = correct / total if total > 0 else 0
            train_loss = running_loss / total if total > 0 else 0
            loop.set_postfix(loss=train_loss, acc=train_acc)

        print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}")


        # Validation with bias correction applied
        val_loss, val_acc, p, r, f1, gil, class_acc = evaluate(
            model, val_loader, device, label_space,
            args.full_model_acc,
            split_name="Val"
        )
        print(f"Epoch {epoch + 1}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        gil_previous = gil_ratio(val_acc, args.full_model_acc)
        print(f"GIL_PreClass-val:{gil_previous:.4f}")

        gil_base = gil_ratio(val_acc, base_model_acc)
        run_log.log("epoch", epoch=epoch + 1, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc,
                    gil_base=gil_base, gil_prev=gil_previous, val_f1=f1)
        print(f"GIL_Base-val:{gil_base:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.checkpoint_dir, f"layoutlmv3_cil_incremental_evm_ood_{args.unseen_classes}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'patience_counter': patience_counter, 
            }, save_path)
            print(f"Saved best model checkpoint: {save_path}")
        else:
            patience_counter += 1
        
        print(f"Patience counter: {patience_counter} / {args.patience}")

        if patience_counter > args.patience:
            print(f"Early stopping triggered. Patience counter exceeded {args.patience}.")
            break
    print("Training completed. Loading best model for testing ...")
    best_path = os.path.join(args.checkpoint_dir, f"layoutlmv3_cil_incremental_evm_ood_{args.unseen_classes}_best.pt")
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    if args.use_bias_correction:
        center_classifier_bias(model)
    model.eval()

    test_loss, test_acc, test_p, test_r, test_f1, test_gil, test_class_acc = evaluate(model, test_loader, device, label_space, args.full_model_acc, split_name="Test")

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

    gil_previous = gil_ratio(test_acc, args.full_model_acc)
    print(f"GIL_PreClass-val:{gil_previous:.4f}")

    gil_base = gil_ratio(test_acc, base_model_acc)
    run_log.log("final", split="Test", loss=test_loss, acc=test_acc, precision=test_p, recall=test_r, f1=test_f1,
                gil_base=gil_base, gil_prev=test_gil, class_acc=test_class_acc)
    print(f"GIL_Base-Test:{gil_base:.4f}")
    
    # ---------- EVM: fitted once per step on its training data (new class + exemplars) with the best model ----------
    fit_evm(evm, model, combined_loader, device, label_space, new_class=unseen_classes[-1])
    # ViM for L_OOD in the next step, fitted on the same training data
    f_tr, z_tr, _ = collect_features_logits(model, combined_loader, device)
    W, b = last_linear_layer(model)
    ood_vim.fit(f_tr, z_tr, W=W, b=b)
    ood_vim.fit_keys = set(label_space)
    if args.evm_persist:
        save_open_set_state(best_path, evm=evm, vim=ood_vim)

    # ---------- EVM open set: classes not learned yet are the unknowns ----------
    future_classes = [c for c in all_classes if c not in label_space]
    ood_test_loader = get_incremental_dataloader(
        dataset_name=args.dataset_name, ocr_tensor_file=args.ocr_tensor_path, classes=future_classes,
        label_classes=all_classes, image_dir=args.data_dir, split="test", batch_size=args.batch_size,
        images_per_class=val_test_img, seed=args.seed,
    ) if future_classes else None
    evm_open_set_eval(evm, model, device, test_loader, label_space, ood_test_loader, all_classes, tag="EVM open set test")
    if ood_test_loader is not None:
        known_train_loader = get_incremental_dataloader(
            dataset_name=args.dataset_name, ocr_tensor_file=args.ocr_tensor_path, classes=label_space,
            label_classes=label_space, image_dir=args.data_dir, split="train", batch_size=args.batch_size,
            images_per_class=args.ood_max_per_class, seed=args.seed,
        )
        evaluate_ood(
            make_detector(args.ood_method), model, device,
            fit_loader=known_train_loader, calib_loader=val_loader, id_loader=test_loader, ood_loader=ood_test_loader,
            tpr=args.ood_tpr, max_per_class=args.ood_max_per_class, tag=f"Open-set (+{unseen_classes[-1]})",
            savepath=os.path.join(args.checkpoint_dir, f"ood_{args.ood_method}_{args.unseen_classes}.png"),
        )

if __name__ == "__main__":
    main()
