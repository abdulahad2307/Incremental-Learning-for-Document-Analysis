import os
import torch
import torch.nn as nn
from typing import List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score

from utils.seed import add_seed_arg, set_seed
from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import get_class_il_loader
from utils.class_IL.train_utils import (
    save_checkpoint, load_checkpoint, train_one_epoch_cil,train_one_epoch_cil_v2, evaluate, CILMetrics
)
from utils.class_IL.cil_utils import (
    StandardIncremental, DistillationIncremental, EWC, ExemplarManager, AdaptiveLR, fill_exemplar_memory
)
from utils.class_IL.training_modes import get_training_mode
from utils import run_log
from utils.eval.predictions import save_predictions
from utils.il_checks import warn_inactive_terms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_unseen_classes(base_classes: List[str], all_classes: List[str]) -> List[str]:
    return [cls for cls in all_classes if cls not in base_classes]


def run_incremental_learning(
    data_root: str,
    ocr_tensor_path: str,
    all_classes: List[str],
    base_classes: List[str],
    unseen_classes: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    strategy: str = "distillation",
    temperature: float = 2.0,
    lambda_distill: float = 1.0,
    lambda_ewc: float = 5000.0,
    use_ewc: bool = True,
    use_exemplars: bool = True,
    joint_training: bool = False,
    max_exemplars: int = 320,
    exemplar_selection: str = "herding",
    training_mode: str = "last_layer",
    trainable_layers: Optional[List[str]] = None,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    global_best_acc: float = 0.0,
    full_model_acc: Optional[float] = None,
    patience: int = 10,
    use_balanced_sampler: bool = False,
    use_bias_correction: bool = False
):
    import shutil
    from torch.utils.data import WeightedRandomSampler, DataLoader
    os.makedirs(checkpoint_dir, exist_ok=True)

    if strategy == "distillation":
        inc_strategy = DistillationIncremental(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)

    exemplar_mgr = ExemplarManager(
        max_exemplars=max_exemplars,
        selection_strategy=exemplar_selection
    ) if use_exemplars else None

    # Validate unseen_classes list
    if unseen_classes is None or len(unseen_classes) == 0:
        raise ValueError("You must provide a non-empty list of unseen_classes")
    for cls in unseen_classes:
        if cls not in all_classes:
            raise ValueError(f"Unseen class '{cls}' is not in all_classes list")
        if cls in base_classes:
            raise ValueError(f"Unseen class '{cls}' is already in base_classes")

    base_model_acc = full_model_acc
    previous_step_best_acc = base_model_acc

    # Variables for resume
    start_unseen_index = 0
    start_epoch = 0
    epochs_no_improve = 0
    step_best_acc = 0.0
    step_best_path = None

    #loading the last checkpoint info
    if resume and resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
        # unseen class index to resume from
        if 'unseen_index' in checkpoint:
            start_unseen_index = checkpoint['unseen_index']
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'epochs_no_improve' in checkpoint:
            epochs_no_improve = checkpoint['epochs_no_improve']
        if 'step_best_acc' in checkpoint:
            step_best_acc = checkpoint['step_best_acc']
        if 'step_best_path' in checkpoint:
            step_best_path = checkpoint['step_best_path']
        if 'current_classes' in checkpoint:
            current_classes = checkpoint['current_classes']
        else:
            current_classes = base_classes.copy()
        del checkpoint
        torch.cuda.empty_cache()
        print(f"Resuming from unseen_index: {start_unseen_index}, epoch: {start_epoch}")

    else:
        current_classes = base_classes.copy()

    ewc = None

    for unseen_idx, new_class in enumerate(unseen_classes[start_unseen_index:], start=start_unseen_index):
        if unseen_idx > start_unseen_index:
            start_epoch = 0
            epochs_no_improve = 0
            step_best_acc = 0.0
            step_best_path = None

        # On resume, current_classes from the checkpoint may already contain new_class
        previous_classes = [c for c in current_classes if c != new_class]
        if new_class not in current_classes:
            current_classes.append(new_class)
        print(f"\n= Incremental Step ({unseen_idx + 1}/{len(unseen_classes)}) - Adding class: {new_class} =")


        new_data_samples = []
        train_dataset = None
        train_loader = None

        from utils.class_IL.dataloader_utils import EAMLClassILDataset, common_transform, eaml_collate_fn

        if model_name == "eaml":
            train_dataset = EAMLClassILDataset(
                data_dir=os.path.join(data_root, "train"),
                current_classes=current_classes,
                ocr_data_path=ocr_tensor_path,
                transform=common_transform
            )
            if not joint_training:
                # Class-incremental data: the new class's samples plus the exemplar memory of all previous classes
                new_data_samples = [s for s in train_dataset.samples if s[2] == new_class]
                replay_samples = []
                if use_exemplars and exemplar_mgr is not None:
                    fill_exemplar_memory(exemplar_mgr, train_dataset, previous_classes, base_model_path,
                                         len(previous_classes), DEVICE)
                    replay_samples = [s for c in previous_classes for s in exemplar_mgr.exemplars.get(c, [])]
                train_dataset.samples = replay_samples + new_data_samples
                print(f"Training on {len(new_data_samples)} new-class samples and {len(replay_samples)} exemplars "
                      f"of {len(previous_classes)} previous classes.")
            else:
                print(f"Joint training on all data of {len(current_classes)} classes (upper-bound baseline).")

            if use_balanced_sampler:
                from collections import Counter
                class_counts = Counter([s[2] for s in train_dataset.samples])
                total = sum(class_counts.values())
                class_weights = {cls: total/count for cls, count in class_counts.items()}
                sample_weights = [class_weights[s[2]] for s in train_dataset.samples]
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
        else:
            train_loader = get_class_il_loader(model_name, os.path.join(data_root, "train"), current_classes,
                                               batch_size, ocr_data=ocr_tensor_path)

        val_loader = get_class_il_loader(model_name, os.path.join(data_root, "val"), current_classes,
                                         batch_size, ocr_data=ocr_tensor_path)
        torch.cuda.empty_cache()

        def build_model(num_classes):
            if model_name == "docformer":
                return DocFormer(DocFormerConfig(), num_classes=num_classes).to(DEVICE)
            return EAMLModel(num_classes=num_classes).to(DEVICE)

        def load_ckpt_into(target, path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            target.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            del ckpt
            torch.cuda.empty_cache()

        resuming_this_step = (resume and resume_checkpoint and os.path.exists(resume_checkpoint)
                              and unseen_idx == start_unseen_index)
        if resuming_this_step:
            # Resume checkpoint already has the expanded head for current_classes
            model = build_model(len(current_classes))
            load_ckpt_into(model, resume_checkpoint)
        elif len(previous_classes) > 0 and os.path.exists(base_model_path):
            # Student starts from the previous step's model; heads are expanded by adapt_model below
            model = build_model(len(previous_classes))
            load_ckpt_into(model, base_model_path)
        else:
            print(f"Warning: no previous checkpoint found at {base_model_path}; student starts from scratch.")
            model = build_model(len(current_classes))

        this_start_epoch = start_epoch if unseen_idx == start_unseen_index else 0
        this_epochs_no_improve = epochs_no_improve if unseen_idx == start_unseen_index else 0
        this_step_best_acc = step_best_acc if unseen_idx == start_unseen_index else 0.0
        this_step_best_path = step_best_path if unseen_idx == start_unseen_index else None

        old_model = None
        if len(previous_classes) > 0:
            # Teacher = previous step's best model (base_model_path is advanced after every step)
            old_model = build_model(len(previous_classes))
            if os.path.exists(base_model_path):
                load_ckpt_into(old_model, base_model_path)
            old_model.eval()
            print(f"prev: {len(previous_classes)}, new: {len(current_classes)}")
            if not resuming_this_step:
                model = inc_strategy.adapt_model(model, len(previous_classes), len(current_classes), model_name).to(DEVICE)

        tm = get_training_mode(model, training_mode, trainable_layers)
        model = tm.prepare_for_training()
        warn_inactive_terms(model, use_ewc=use_ewc)
        optimizer = torch.optim.AdamW(
            tm.get_trainable_params(),
            lr=lr,
            weight_decay=weight_decay
        )

        # Loading optimizer state if resuming
        if resuming_this_step:
            checkpoint = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            del checkpoint
            torch.cuda.empty_cache()

        lr_sched = AdaptiveLR(optimizer, base_lr=lr)
        criterion = nn.CrossEntropyLoss()

        # ----------- TRAINING -----------
        for epoch in range(this_start_epoch, num_epochs):
            print(f"\n= Epoch {epoch+1}/{num_epochs} for Class {new_class} =")
            train_metrics = CILMetrics(current_classes)
            train_result = train_one_epoch_cil_v2(
                model,
                train_loader,
                optimizer,
                criterion,
                DEVICE,
                train_metrics,
                inc_strategy=inc_strategy,
                old_model=old_model,
                ewc=ewc if use_ewc else None
            )
            val_metrics = CILMetrics(current_classes)
            val_result = evaluate(model, val_loader, DEVICE, val_metrics, None)
            print(f"Train Loss: {train_result['loss']:.4f} | Train Acc: {train_result['top1_acc']:.4f}")
            print(f"Val Loss: {val_result['loss']:.4f} | Val Acc: {val_result['top1_acc']:.4f}")
            run_log.log("epoch", step=unseen_idx + 1, new_class=new_class, epoch=epoch + 1,
                        train_loss=train_result['loss'], train_acc=train_result.get('accuracy', train_result.get('top1_acc')),
                        val_loss=val_result['loss'], val_acc=val_result['top1_acc'])

            last_epoch_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}_{new_class}.pth")
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
                                })
            else:
                this_epochs_no_improve += 1
                print(f"Patience counter: {this_epochs_no_improve}/{patience}")
                save_checkpoint(model, optimizer, epoch + 1,
                                last_epoch_path,
                                extra_data={
                                    "current_classes": current_classes,
                                    "unseen_index": unseen_idx,
                                    "unseen_class": new_class,
                                    "epoch": epoch + 1,
                                    "epochs_no_improve": this_epochs_no_improve,
                                    "step_best_acc": this_step_best_acc,
                                    "step_best_path": this_step_best_path
                                })
                if this_epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} due to no improvement in val accuracy for {patience} epochs.")
                    break

            lr_sched.step(val_result['top1_acc'])
            torch.cuda.empty_cache()

        # End-of-step updates (EWC, exemplars, EVM) use the step's best model, which is what the next step continues from
        if this_step_best_path and os.path.exists(this_step_best_path):
            load_ckpt_into(model, this_step_best_path)

        # ---------- EWC ----------
        if use_ewc:
            print(f"Updating EWC Fisher information for increment {unseen_idx + 1}...")
            ewc = EWC(model, train_loader, DEVICE, lambda_ewc)

        # ---------- EXEMPLAR UPDATE ----------
        if use_exemplars and exemplar_mgr is not None:
            exemplar_mgr.update(train_loader.dataset, new_class, model)
            for cls in exemplar_mgr.exemplars:
                print(f"Exemplar samples for {cls}: {len(exemplar_mgr.exemplars[cls])}")

        # ---------- BIAS CORRECTION ----------
        if use_bias_correction:
            if hasattr(model, "fusion_classifier"):
                print("Applying bias correction to last (fusion) classifier.")
                with torch.no_grad():
                    mean_bias = model.fusion_classifier.bias.mean().item()
                    model.fusion_classifier.bias[:] -= mean_bias

        if this_step_best_path and os.path.exists(this_step_best_path):
            base_model_path = this_step_best_path
        else:
            print("Warning: No best checkpoint found for this increment. Skipping base_model_path update.")

    # After all increments done
    final_global_best_path = None
    if this_step_best_path and os.path.exists(this_step_best_path):
        final_global_best_path = os.path.join(checkpoint_dir, "final_global_best_model.pth")
        shutil.copy(this_step_best_path, final_global_best_path)
        print(f"Final Global Best Model saved: {final_global_best_path}")
        torch.cuda.empty_cache()
    else:
        print("Warning: No best checkpoint found for the last incremental step. Not saving final global best model.")

    if final_global_best_path:
        learned_classes = list(current_classes)  # the final model has one output per class learned so far
        print("\n= Final Evaluation: GLOBAL BEST MODEL on Full Datasets =")
        if model_name == "docformer":
            cfg = DocFormerConfig()
            final_model = DocFormer(cfg, num_classes=len(learned_classes)).to(cfg.device)
        else:
            final_model = EAMLModel(num_classes=len(learned_classes)).to(DEVICE)

        checkpoint = torch.load(final_global_best_path, map_location=DEVICE,weights_only=False)
        final_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        del checkpoint
        torch.cuda.empty_cache()
        final_model.eval()
        if use_bias_correction and hasattr(final_model, "fusion_classifier"):
            # Bias correction on the evaluated model: the step checkpoint holds the model before the correction
            with torch.no_grad():
                final_model.fusion_classifier.bias -= final_model.fusion_classifier.bias.mean()

        train_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "train"),
                                                learned_classes, batch_size, ocr_data=ocr_tensor_path)
        val_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "val"),
                                              learned_classes, batch_size, ocr_data=ocr_tensor_path)
        test_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "test"),
                                               learned_classes, batch_size, ocr_data=ocr_tensor_path)

        def evaluate_with_metrics(split_name, loader):
            eval_metrics_eval = CILMetrics(class_names=learned_classes.copy())
            results = evaluate(final_model, loader, DEVICE, eval_metrics_eval, full_model_acc)
            loss, acc = results.get('loss', 0), results.get('top1_acc', 0)
            preds, labels = results.get('preds', []), results.get('labels', [])
            p = precision_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            r = recall_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            f1 = f1_score(labels, preds, average='macro', zero_division=0) if len(labels) > 0 else 0
            gil = (acc - full_model_acc) / (1 - full_model_acc) if full_model_acc is not None else None
            print(f"{split_name} - Loss: {loss:.4f} | Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")
            run_log.log("final", split=split_name, loss=loss, acc=acc, precision=p, recall=r, f1=f1, gil_base=gil,
                        class_acc=dict(zip(learned_classes, results['class_acc'])) if split_name == "Test" and results.get('class_acc') is not None else None)
            if gil is not None:
                print(f"{split_name} - Incremental Learning Gap (G_IL): {gil:.4f}")
            if split_name == "Test":
                class_wise_acc = results.get('class_acc', None)
                if class_wise_acc is not None:
                    print("\nClass-wise Test Accuracy:")
                    for clss, cacc in zip(learned_classes, class_wise_acc):
                        print(f"  {clss}: {cacc:.4f}")
            torch.cuda.empty_cache()

        evaluate_with_metrics("Train", train_loader_full)
        evaluate_with_metrics("Validation", val_loader_full)
        evaluate_with_metrics("Test", test_loader_full)

        # Per-document predictions of this step: test documents of the learned and of the not-yet-learned classes
        future_classes = [c for c in all_classes if c not in learned_classes]
        future_loader = get_class_il_loader(model_name, os.path.join(data_root, "test"), future_classes, batch_size,
                                            ocr_data=ocr_tensor_path) if future_classes else None
        save_predictions(final_model, DEVICE, checkpoint_dir, f"cil_{new_class}", learned_classes,
                         {"seen": (test_loader_full, learned_classes), "unseen": (future_loader, future_classes)},
                         evm=None, ood_detector=None, new_class=new_class, checkpoint=final_global_best_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_path', required=True)
    p.add_argument('--all_classes', required=True)
    p.add_argument('--base_classes', required=True)
    p.add_argument('--unseen_classes', required=True, help="Unseen class list for this incremental step")
    p.add_argument('--base_model_path', required=True)
    p.add_argument('--model_name', choices=['eaml', 'docformer'], required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--use_bias_correction', action='store_true')
    p.add_argument('--use_balanced_sampler', action='store_true')
    p.add_argument('--joint_training', action='store_true', help='Train each step on all data of all seen classes (upper-bound baseline)')
    p.add_argument('--max_exemplars', type=int, default=200)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='herding')
    p.add_argument('--training_mode', choices=['classifier_only','last_layer','full','full_model','selective'], default='last_layer')
    p.add_argument('--trainable_layers', nargs='+', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume_checkpoint', type=str, default=None)
    p.add_argument('--global_best_acc', type=float, default=0.0)
    p.add_argument('--full_model_acc', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=0.01)
    #p.add_argument('--test_interval', type=int, default=20)
    p.add_argument('--patience', type=int, default=10)
    add_seed_arg(p)
    args = p.parse_args()
    set_seed(args.seed)
    run_log.init("eaml", "CIL", "No EVM", args)

    run_incremental_learning(
        data_root=args.data_dir,
        ocr_tensor_path=args.ocr_tensor_path,
        all_classes=args.all_classes.split(','),
        base_classes=args.base_classes.split(','),
        unseen_classes=args.unseen_classes.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill,
        lambda_ewc=args.lambda_ewc,
        use_ewc=args.use_ewc,
        use_exemplars=args.use_exemplars,
        use_bias_correction=args.use_bias_correction,
        use_balanced_sampler=args.use_balanced_sampler,
        joint_training=args.joint_training,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        training_mode=args.training_mode,
        trainable_layers=args.trainable_layers,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        global_best_acc=args.global_best_acc,
        full_model_acc=args.full_model_acc,
        weight_decay=args.weight_decay,
        patience=args.patience
    )