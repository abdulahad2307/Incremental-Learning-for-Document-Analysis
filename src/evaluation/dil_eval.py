import torch
import argparse
import time
from torch.nn import functional as F

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig

from utils.domain_IL.dataloader_utils import get_dataloaders

@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    correct = total = 0

    for batch in dataloader:
        images = batch.get('images') or batch.get('pixel_values')
        labels = batch['labels']
        images = images.to(device)
        labels = labels.to(device)

        if 'texts' in batch:
            input_ids = batch['texts']['input_ids'].to(device)
            attention_mask = batch['texts']['attention_mask'].to(device)
            outputs = model(images, input_ids=input_ids, attention_mask=attention_mask)
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bboxes = batch['bboxes'].to(device)
            outputs = model(images, input_ids=input_ids, attention_mask=attention_mask, bboxes=bboxes)

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

@torch.no_grad()
def evaluate_ensemble(models, dataloader, device):
    for m in models:
        m.eval()
    correct = total = 0

    for batch in dataloader:
        images = batch.get('images') or batch.get('pixel_values')
        labels = batch['labels']
        images = images.to(device)
        labels = labels.to(device)

        probs = []
        if 'texts' in batch:
            input_ids = batch['texts']['input_ids'].to(device)
            attention_mask = batch['texts']['attention_mask'].to(device)
            for model in models:
                out = model(images, input_ids=input_ids, attention_mask=attention_mask)
                probs.append(F.softmax(out, dim=1))
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            bboxes = batch['bboxes'].to(device)
            for model in models:
                out = model(images, input_ids=input_ids, attention_mask=attention_mask, bboxes=bboxes)
                probs.append(F.softmax(out, dim=1))

        avg_prob = sum(probs) / len(probs)
        preds = torch.argmax(avg_prob, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    models = []
    if args.model in ["eaml", "both"]:
        eaml_model = EAMLModel(num_classes=args.num_classes).to(device)
        eaml_model.load_state_dict(torch.load(args.eaml_ckpt, map_location=device)['model_state_dict'])
        models.append(eaml_model)

    if args.model in ["docformer", "both"]:
        config = DocFormerConfig()
        docformer_model = DocFormer(config,num_classes=args.num_classes).to(device)
        docformer_model.load_state_dict(torch.load(args.doc_ckpt, map_location=device)['model_state_dict'])
        models.append(docformer_model)

    loaders = get_dataloaders(
        model_type="eaml" if args.model == "eaml" else "docformer",
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        mode="test"
    )

    test_loader = loaders["test"]

    print("Evaluating...")
    start = time.time()

    if args.model == "both" and args.ensemble:
        acc = evaluate_ensemble(models, test_loader, device)
        print(f"Ensemble Accuracy: {acc:.4f}")
    else:
        acc = evaluate_model(models[0], test_loader, device)
        print(f"{args.model.upper()} Accuracy: {acc:.4f}")

    print(f"Finished in {time.time() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Post-DIL Models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", choices=["eaml", "docformer", "both"], default="both")
    parser.add_argument("--eaml_ckpt", type=str, default="./checkpoints_dil/eaml_dil.pth")
    parser.add_argument("--doc_ckpt", type=str, default="./checkpoints_dil/docformer_dil.pth")
    parser.add_argument("--ensemble", action="store_true", help="Enable ensemble evaluation")
    parser.add_argument("--num_classes", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    main(args)
