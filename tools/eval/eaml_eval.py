import torch
import json
from utils.eaml.dataloader import EAML_DataLoader
from utils.eaml.eaml_model import EAMLModel
from utils.eaml.mutual_learning import MutualLearningLoss

def evaluate_model_on_test():
    # Hardcoded paths and parameters
    model_checkpoint_path= "/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_with_KDILS_Final/best_model_specification.pth"

    data_dir = "/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
    ocr_data_path = "/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.pt"  # precomputed OCR file
    #all_class_list = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo', 'scientific_publication']
    all_class_list = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report', 'scientific_publication', 'specification', 'file_folder', 'news_article', 'budget', 'invoice', 'presentation', 'questionnaire', 'resume', 'memo']

    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    # Initialize DataLoader for test data
    eaml_loader = EAML_DataLoader(
        data_dir=data_dir,
        batch_size=batch_size,
        class_list=all_class_list,
        ocr_data_path=ocr_data_path
    )
    test_loader = eaml_loader.get_loader('test', shuffle=False)

    # Initialize model
    model = EAMLModel(
        num_classes=len(all_class_list),
        embed_dim=512,    # Must match what was used in training
        freeze_image_encoder=False
    )
    model.to(device)

    # Load the trained model checkpoint
    if not torch.cuda.is_available():
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu',weights_only=False)
    else:
        checkpoint = torch.load(model_checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Criterion for evaluation (same as used in training)
    criterion = MutualLearningLoss(cls_weight=1.0, kld_weight=0.3, threshold=0.1)

    correct = 0
    total = 0
    total_loss = 0.0

    # For class-wise accuracy
    class_correct = [0 for _ in range(len(all_class_list))]
    class_total = [0 for _ in range(len(all_class_list))]

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                images = batch['images'].to(device)
                texts = {
                    'input_ids': batch['texts']['input_ids'].to(device),
                    'attention_mask': batch['texts']['attention_mask'].to(device)
                }
                labels = batch['labels'].to(device)
            except Exception as e:
                print(f"Warning: Skipping corrupted batch {batch_idx} due to loading error: {e}")
                continue  # Skip this batch and continue

            outputs = model(images, texts, return_features=True)
            loss_dict = criterion(outputs, labels)
            total_loss += loss_dict['total_loss'].item()
            _, predicted = torch.max(outputs['fusion_logits'].data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Class-wise accuracy update
            for i in range(labels.size(0)):
                label_i = labels[i].item()
                pred_i = predicted[i].item()
                class_total[label_i] += 1
                if label_i == pred_i:
                    class_correct[label_i] += 1

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    print(f"Test evaluation completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("\nClass-wise accuracy-1stInc:")
    for idx, cls_name in enumerate(all_class_list):
        if class_total[idx] > 0:
            acc = 100 * class_correct[idx] / class_total[idx]
            print(f"{cls_name}: {acc:.2f}% ({class_correct[idx]}/{class_total[idx]})")
        else:
            print(f"{cls_name}: No samples")

if __name__ == "__main__":
    evaluate_model_on_test()
