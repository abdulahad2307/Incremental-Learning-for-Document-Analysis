import os
import random
import torch
import torch.nn as nn
import numpy as np

from utils.seed import add_seed_arg, set_seed
from utils.domain_IL.dil_dataloader import DILDataLoader
from utils.domain_IL.dil_train_utils import (
    save_checkpoint_dil, save_epoch_checkpoint_dil, train_one_epoch_dil,
    evaluate_dil, classwise_accuracy, evaluate_domain
)
from utils.domain_IL.dil_utils import (
    StandardDomainIL, DistillationDomainIL, EWC, ExemplarManager, AdaptiveLR, extract_features_by_class, mix_replay
)
from utils.domain_IL.dil_model_loader import load_eaml_model_partial, set_finetune_mode
from utils import run_log

from utils.evm.evm_classifier import EVMClassifier
from utils.evm.evm_eval import evm_openset_metrics
from utils.evm.evm_viz import plot_openset_histograms
from utils.ood.ood_eval import make_detector, evaluate_ood, subsample_loader, collect_features_logits, last_linear_layer
from utils.ood.vim import VIM_OOD
from utils.ood.ood_loss import vim_ood_loss
from utils.evm.evm_loss import evm_nll_loss
from utils.il_checks import warn_inactive_terms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_domain_classwise(model, loader, global_classes, domain_name, device):
    acc = evaluate_dil(model, {domain_name: loader}, device)
    class_acc = classwise_accuracy(model, loader, device, len(global_classes))
    print(f"\nDomain: {domain_name} - Overall Acc: {acc:.4f}")
    for i, cacc in enumerate(class_acc):
        print(f"  Class {global_classes[i]}: {cacc:.4f}")
    return acc, class_acc

def train_one_epoch_dil_with_evm_ood(
    model,
    train_loaders,
    optimizer,
    criterion,
    device,
    strategy,
    exemplar_manager=None,
    ewc=None,
    old_model=None,
    use_bias_correction=True,
    evm=None,
    ood_detector=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for domain_name, train_loader in train_loaders.items():
        for batch in train_loader:
            optimizer.zero_grad()

            # Current-domain batch plus replayed exemplars of the pretrained domain (if a replay memory is used)
            images, labels, texts = mix_replay(exemplar_manager, batch["images"], batch["labels"], batch.get("texts"), device)
            input_ids = texts["input_ids"]
            attention_mask = texts["attention_mask"]
            batch = {"images": images, "labels": labels, "texts": texts}

            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            features = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask, texts=texts)

            inputs_for_ood = {
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            # EVM loss over samples whose class the EVM has been fitted on (EVM keys are global label ids)
            extra_loss = 0.0
            evm_loss = evm_nll_loss(evm, features, labels)
            if evm_loss is not None:
                extra_loss = lambda_evm * evm_loss
            # L_OOD (ViM virtual-logit penalty) over samples of pretrained-domain classes
            ood_loss = vim_ood_loss(ood_detector, features, logits, labels)
            if ood_loss is not None:
                extra_loss = extra_loss + lambda_ood * ood_loss

            # Incremental strategy loss (already includes the cross-entropy)
            if strategy:
                inc_loss, preds, target_labels = strategy.compute_loss(
                    model, batch, criterion, old_model=old_model, ewc=ewc
                )
                loss = inc_loss + extra_loss
            else:
                loss = criterion(logits, labels) + extra_loss
                preds = torch.argmax(logits, dim=1)
                target_labels = labels

            loss.backward()
            optimizer.step()

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(target_labels.detach().cpu().tolist())
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc

def run_domain_incremental_evm_ood(
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
    resume=False,
    resume_ckpt_path=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
    ood_method="msp",
    ood_tpr=0.95,
    ood_max_per_class=500,
    evm_tailsize=0.3,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(global_classes)}

    inc_strategy = (
        DistillationDomainIL(DEVICE, temperature, lambda_distill)
        if strategy == "distillation" else StandardDomainIL(DEVICE)
    )

    exemplar_mgr = (
        ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection)
        if use_exemplars else None
    )

    ood_detector = make_detector(ood_method)
    evm = EVMClassifier(tailsize=evm_tailsize, cover_threshold=0.7)

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
    val_loaders = dil_loader.get_domain_loaders('val')
    val_loader = val_loaders.get(incremental_domain)
    val_loader_pretrained = val_loaders.get(pretrained_domain)

    test_loaders = dil_loader.get_domain_loaders('test')
    test_loader_pretrained = test_loaders.get(pretrained_domain)
    test_loader_incremental = test_loaders.get(incremental_domain)

    old_classes = 16
    new_classes = len(global_classes)

    model = load_eaml_model_partial(
        eaml_ckpt_path, old_classes, new_classes, device=DEVICE, text_branch=True
    )
    model = set_finetune_mode(model, finetune_mode, unfreeze_depth)
    warn_inactive_terms(model, use_ewc=use_ewc, lambda_evm=lambda_evm, lambda_ood=lambda_ood)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay
    )
    lr_scheduler = AdaptiveLR(optimizer)
    criterion = nn.CrossEntropyLoss()

    # Fisher from the pretrained domain's val split (not test) to avoid test leakage
    ewc = EWC(model, dil_loader.get_domain_loaders('val').get(pretrained_domain), DEVICE, lambda_ewc) if use_ewc else None

    old_model = load_eaml_model_partial(
        eaml_ckpt_path, old_classes, old_classes, device=DEVICE, text_branch=True
    )
    old_model.eval()  # teacher keeps its original old_classes outputs; distillation covers those classes

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    no_improve = 0
    start_epoch = 1

    if resume and resume_ckpt_path is not None and os.path.exists(resume_ckpt_path):
        print(f"Resuming from: {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        no_improve = checkpoint.get('no_improve', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    ood_vim = VIM_OOD()
    if lambda_ood > 0:
        # ViM behind L_OOD: principal subspace of the pretrained domain, fitted with the base model before adaptation
        rvl_train = dil_loader.get_domain_loaders('train').get(pretrained_domain)
        f_tr, z_tr, y_tr = collect_features_logits(model, subsample_loader(rvl_train, ood_max_per_class), DEVICE)
        W, b = last_linear_layer(model)
        ood_vim.fit(f_tr, z_tr, W=W, b=b)
        ood_vim.fit_keys = set(int(y) for y in y_tr)

    if exemplar_mgr is not None:
        # Replay memory: exemplars of the pretrained domain, selected with the base model before adaptation
        exemplar_mgr.update_exemplars(model, dil_loader.get_domain_loaders('train').get(pretrained_domain).dataset, DEVICE)

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loader_dict = {incremental_domain: train_loader}
        train_loss, train_acc = train_one_epoch_dil_with_evm_ood(
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
            evm=evm,
            ood_detector=ood_vim,
            lambda_evm=lambda_evm,
            lambda_ood=lambda_ood,
        )

        val_loss_tob, val_acc_tob = evaluate_dil(model, {incremental_domain: val_loader}, DEVICE)
        val_loss_rvl, val_acc_rvl = evaluate_dil(model, {pretrained_domain: val_loader_pretrained}, DEVICE)
        # model selection on the mean of both domains' val accuracy (same as the LayoutLMv3 DIL scripts)
        val_loss, val_acc = (val_loss_rvl + val_loss_tob) / 2, (val_acc_rvl + val_acc_tob) / 2
        print(f"Val Acc RVL-CDIP: {val_acc_rvl:.4f}, Tobacco-3482: {val_acc_tob:.4f}")

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
            save_checkpoint_dil(model, optimizer, epoch, best_model_path, extra_data=extra_data)
        save_epoch_checkpoint_dil(model, optimizer, epoch, checkpoint_dir, is_best=is_best, extra_data=extra_data, max_keep_last=2)

        if is_best:
            print(f"Validation improved at epoch {epoch}, evaluating test sets and OOD/EVM metrics...")
            # Fit EVM on training features
            train_feats = extract_features_by_class(model, train_loader, DEVICE)
            """
            all_features = []
            model.eval()
            with torch.no_grad():
                for batch in train_loader:
                    images = batch["images"].to(DEVICE)
                    input_ids = batch["texts"]["input_ids"].to(DEVICE)
                    attention_mask = batch["texts"]["attention_mask"].to(DEVICE)
                    feats = model.extract_features_by_class(
                        images=images, input_ids=input_ids, attention_mask=attention_mask, texts=batch["texts"]
                    )
                    all_features.append(feats.cpu())
            train_feats = torch.cat(all_features, dim=0)
            """
            evm = evm.fit(train_feats)
            #plot_ood_histograms(ood_res["y_true"], ood_res["y_pred"], ood_res["scores"], savepath=os.path.join(checkpoint_dir, f"ood_hist_epoch{epoch}.png"))
            #plot_openset_histograms(evm,val_feats,evm_res["scores"],savepath=os.path.join(checkpoint_dir, f"evm_hist_epoch{epoch}.png"))

            for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
                acc, acc_cls = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
                run_log.log("step_test", epoch=epoch, split=domain_name, acc=acc, **{("acc_rvl" if domain_name == pretrained_domain else "acc_tob"): acc},
                            class_acc=dict(zip(global_classes, acc_cls)))
                print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")

        lr_scheduler.step({'accuracy': val_acc, 'loss': val_loss})

    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=False)['model_state_dict'])
    print("Final evaluation on test datasets:")

    for domain_name, loader in [(pretrained_domain, test_loader_pretrained), (incremental_domain, test_loader_incremental)]:
        acc, acc_cls = evaluate_domain(model, loader, DEVICE, len(global_classes), global_classes)
        run_log.log("final", split=domain_name, acc=acc, **{("acc_rvl" if domain_name == pretrained_domain else "acc_tob"): acc},
                    class_acc=dict(zip(global_classes, acc_cls)))
        print(f"Test accuracy on domain '{domain_name}': {acc:.4f}")

    # EVM open set: the EVM is fitted on incremental-domain training classes, so classes it never saw
    # (e.g. RVL-CDIP-only classes) are its unknowns. Evaluated on both domains' test sets.
    if evm.initialized:
        test_feats = {}
        for loader in (test_loader_pretrained, test_loader_incremental):
            for cls, feats in extract_features_by_class(model, subsample_loader(loader, ood_max_per_class), DEVICE).items():
                test_feats[cls] = np.vstack([test_feats[cls], feats]) if cls in test_feats else feats
        evm_test_res = evm_openset_metrics(evm, test_feats)
        run_log.log("open_set", split="test (final)", evm_known_acc=evm_test_res['open_set_accuracy'],
                    evm_unknown_rej=evm_test_res['unknown_rejection'], n_known=evm_test_res['n_known'],
                    n_unknown=evm_test_res['n_unknown'])
        print(f"[EVM test] known acc {evm_test_res['open_set_accuracy']:.4f}, unknown rejection {evm_test_res['unknown_rejection']:.4f} "
              f"(n_known={evm_test_res['n_known']}, n_unknown={evm_test_res['n_unknown']})")
        plot_openset_histograms(evm_test_res["y_true"], evm_test_res["y_pred"], evm_test_res["scores"],
                                savepath=os.path.join(checkpoint_dir, "evm_hist_final.png"), known_mask=evm_test_res["is_known"])

    # Domain-shift detection: pretrained domain = in-distribution, incremental domain = OOD,
    # for the base model (before adaptation) and the adapted best model.
    base_model = load_eaml_model_partial(eaml_ckpt_path, old_classes, new_classes, device=DEVICE, text_branch=True)
    for label, eval_model in [("before adaptation", base_model), ("after adaptation", model)]:
        evaluate_ood(
            ood_detector, eval_model, DEVICE,
            fit_loader=dil_loader.get_domain_loaders('train').get(pretrained_domain),
            calib_loader=dil_loader.get_domain_loaders('val').get(pretrained_domain),
            id_loader=test_loader_pretrained,
            ood_loader=test_loader_incremental,
            tpr=ood_tpr, max_per_class=ood_max_per_class,
            tag=f"Domain shift {pretrained_domain} -> {incremental_domain}, {label}",
            savepath=os.path.join(checkpoint_dir, f"ood_{ood_method}_domain_shift_{label.split()[0]}.png"),
        )


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
    p.add_argument('--lambda_evm', type=float, default=0.1)
    p.add_argument('--lambda_ood', type=float, default=0.1)
    p.add_argument('--ood_method', type=str, default='msp', choices=['msp', 'vim', 'gradnorm'])
    p.add_argument('--ood_tpr', type=float, default=0.95, help='Fraction of in-distribution val samples accepted; sets the OOD threshold')
    p.add_argument('--ood_max_per_class', type=int, default=500, help='Samples per class used for OOD fitting/calibration/testing')
    p.add_argument('--ood_threshold', type=float, default=None, help='Deprecated, ignored: the threshold is calibrated with --ood_tpr')
    p.add_argument('--evm_tailsize', type=float, default=0.3)

    add_seed_arg(p)
    args = p.parse_args()
    set_seed(args.seed)
    run_log.init("eaml", "DIL", "EVM+OOD", args)
    run_domain_incremental_evm_ood(
        data_root=args.data_dir,
        ocr_tensor_dirs={d: t for d, t in zip(args.domains.split(','), args.ocr_tensor_dirs)},
        domains=args.domains.split(','),
        #global_classes=args.global_classes.split(','),
        global_classes = [c.strip().replace('{','').replace('}','') for c in args.global_classes.split(',')],
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
        resume=args.resume,
        resume_ckpt_path=args.resume_ckpt_path,
        lambda_evm=args.lambda_evm,
        lambda_ood=args.lambda_ood,
        ood_method=args.ood_method,
        ood_tpr=args.ood_tpr,
        ood_max_per_class=args.ood_max_per_class,
        evm_tailsize=args.evm_tailsize
    )
