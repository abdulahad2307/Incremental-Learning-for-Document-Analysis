import os
import random
import torch
import torch.nn as nn
import numpy as np
#from sklearn.metrics import precision_score, recall_score, f1_score
#from utils.eaml.eaml_model import EAMLModel
from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_train_utils import (
    save_checkpoint_dil, save_epoch_checkpoint_dil, train_one_epoch_dil,
    evaluate_dil, classwise_accuracy, evaluate_domain
)
from utils.domain_IL.dil_utils import (
    StandardDomainIL, DistillationDomainIL, EWC, ExemplarManager, AdaptiveLR
)
#from utils.domain_IL.training_modes import get_dil_training_mode
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode
from utils import run_log
from utils.il_checks import warn_inactive_terms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_domain_classwise(model, loader, global_classes, domain_name, device):
    acc = evaluate_dil(model, {domain_name: loader}, device)
    class_acc = classwise_accuracy(model, loader, device, len(global_classes))
    print(f"\nDomain: {domain_name} - Overall Acc: {acc:.4f}")
    for i, cacc in enumerate(class_acc):
        print(f"  Class {global_classes[i]}: {cacc:.4f}")
    return acc, class_acc


def run_domain_incremental(
    data_root,
    ocr_tensor_dirs,
    domains,
    global_classes,
    eaml_ckpt_path,
    checkpoint_dir,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    num_epochs=10,
    strategy="distillation",
    temperature=2.0,
    lambda_distill=1.0,
    lambda_ewc=5000.0,
    use_ewc=True,
    use_exemplars=False,
    max_exemplars=200,
    exemplar_selection="random",
    finetune_mode="last_layer",
    unfreeze_depth=0,
    patience=10,
    use_bias_correction=True,
    resume =False,
    resume_ckpt_path=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

    inc_strategy = DistillationDomainIL(DEVICE, temperature, lambda_distill) if strategy == "distillation" else StandardDomainIL(DEVICE)
    exemplar_mgr = ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection) if use_exemplars else None

    dil_loader = DILDataLoader(
        data_root=data_root,
        domain_list=domains,
        batch_size=batch_size,
        img_size=(229, 229),
        num_workers=4,
        ocr_tensor_dirs=ocr_tensor_dirs,
        class_to_idx=class_to_idx
    )

    pretrained_domain = domains[0]
    incremental_domain = domains[1]

    train_loader = dil_loader.get_domain_loaders('train').get(incremental_domain)
    val_loader = dil_loader.get_domain_loaders('val').get(incremental_domain)

    test_loaders = dil_loader.get_domain_loaders('test')
    test_loader_pretrained = test_loaders.get(pretrained_domain)
    test_loader_incremental = test_loaders.get(incremental_domain)

    old_classes = 16
    new_classes = len(global_classes)

    model = load_eaml_model_partial(
        eaml_ckpt_path,
        old_classes,
        new_classes,
        device=DEVICE,
        text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)
    warn_inactive_terms(model, use_ewc=use_ewc)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    lr_scheduler = AdaptiveLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()

    # Fisher from the pretrained domain's val split (not test) to avoid test leakage
    ewc = EWC(model, dil_loader.get_domain_loaders('val').get(pretrained_domain), DEVICE, lambda_ewc) if use_ewc else None

    old_model = load_eaml_model_partial(
        eaml_ckpt_path,
        old_classes,
        old_classes,
        device=DEVICE,
        text_branch=True
    )
    old_model.eval()  # teacher keeps its original old_classes outputs; distillation covers those classes

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = None
    no_improve = 0
    start_epoch = 1

    # Resume checkpoint loading if provided
    if resume is True and resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        no_improve = checkpoint.get('no_improve', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        print(f"Resumed at epoch {start_epoch}, patience={no_improve}, best_val_acc={best_val_acc}, best_val_loss={best_val_loss}")


    print("Dataset overview:")
    counts = dil_loader.get_class_counts()
    for domain, count in counts.items():
        print(f"  Domain '{domain}': {count} classes")

    print("=== Training Starting ===")
    if exemplar_mgr is not None:
        # Replay memory: exemplars of the pretrained domain, selected with the base model before adaptation
        exemplar_mgr.update_exemplars(model, dil_loader.get_domain_loaders('train').get(pretrained_domain).dataset, DEVICE)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loader_dict = {incremental_domain: train_loader}

        train_args = dict(
            model=model,
            train_loaders=train_loader_dict,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            strategy=inc_strategy,
            exemplar_manager=exemplar_mgr,
            ewc=ewc,
            old_model=old_model if isinstance(inc_strategy, DistillationDomainIL) else None,
            use_bias_correction=use_bias_correction,
        )
        train_loss, train_acc = train_one_epoch_dil(**train_args)

        val_loss, val_acc = evaluate_dil(model, {incremental_domain: val_loader}, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        run_log.log("epoch", epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)

        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            print(f"Patience counter: {no_improve}/{patience}")
            if no_improve >= patience:
                print("Early stopping triggered")
                break

        extra_data = {
            "no_improve": no_improve,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
        }
        if is_best:
            save_checkpoint_dil(model, optimizer, epoch, os.path.join(checkpoint_dir, "best_model.pth"), extra_data=extra_data)
        save_epoch_checkpoint_dil(model, optimizer, epoch, checkpoint_dir, is_best=is_best, extra_data=extra_data, max_keep_last=2)

        if is_best:
            print(f"Validation improved at epoch {epoch}, evaluating test sets...")
            pretrained_acc, pretrained_acc_cls = evaluate_domain(model, test_loader_pretrained, DEVICE, len(global_classes), global_classes)
            print(f"Test accuracy on pretrained domain '{pretrained_domain}': {pretrained_acc:.4f}")

            incremental_acc, incremental_acc_cls = evaluate_domain(model, test_loader_incremental, DEVICE, len(global_classes), global_classes)
            run_log.log("step_test", epoch=epoch, split="test", acc_rvl=pretrained_acc, acc_tob=incremental_acc,
                        class_acc_rvl=dict(zip(global_classes, pretrained_acc_cls)),
                        class_acc_tob=dict(zip(global_classes, incremental_acc_cls)))
            print(f"Test accuracy on incremental domain '{incremental_domain}': {incremental_acc:.4f}")

        lr_scheduler.step({'accuracy': val_acc, 'loss': val_loss})

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False)['model_state_dict'])
    print("Final evaluation on test datasets:")

    for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
        acc, acc_cls = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
        run_log.log("final", split=domain_name, acc=acc, **{("acc_rvl" if domain_name == pretrained_domain else "acc_tob"): acc},
                    class_acc=dict(zip(global_classes, acc_cls)))
        print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")





if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_dirs', nargs=2, required=True, help='Two tensor dirs matching domains')
    p.add_argument('--domains', required=True, help='Comma-separated domain names')
    p.add_argument('--global_classes', required=True, help='Comma-separated global class names')
    p.add_argument('--eaml_ckpt_path', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--max_exemplars', type=int, default=200)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='random')
    p.add_argument('--finetune_mode', choices=['head_only', 'last_layer', 'partial_finetune', 'full_finetune'], default='last_layer')
    p.add_argument('--unfreeze_depth', type=int, default=0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--use_bias_correction', action='store_true')
    p.add_argument('--resume', action='store_true', help="Resume training from last checkpoint.")
    p.add_argument('--resume_ckpt_path', type=str)

    args = p.parse_args()
    run_log.init("eaml", "DIL", "No EVM", args)
    run_domain_incremental(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(args.domains.split(','), args.ocr_tensor_dirs)},
        domains=args.domains.split(','),
        global_classes=args.global_classes.split(','),
        eaml_ckpt_path=args.eaml_ckpt_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=0.0,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill,
        lambda_ewc=args.lambda_ewc,
        use_ewc=args.use_ewc,
        use_exemplars=args.use_exemplars,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        finetune_mode=args.finetune_mode,
        unfreeze_depth=args.unfreeze_depth,
        patience=args.patience,
        use_bias_correction=args.use_bias_correction,
        resume = args.resume,
        resume_ckpt_path = args.resume_ckpt_path
    )
