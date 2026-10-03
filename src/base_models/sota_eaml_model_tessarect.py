import torch
import torch.optim as optim
import argparse
import os
import datetime
import pandas as pd
from tqdm import tqdm
from utils.dataloader_tessarect import DataLoader
from utils.eaml.eaml_model import EAMLModel
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

OUTPUT_DIR = "/home/woody/iwi5/iwi5280h/saved_models"

def train(model, loaders, epochs, optimizer, criterion, device):
    """
    Train the EAML model and validate at each epoch.

    Args:
        model: The model instance.
        loaders: Dictionary containing 'train' and 'val' DataLoaders.
        epochs: Number of training epochs.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to use ('cuda' or 'cpu').
    
    Returns:
        Best model based on validation accuracy.
    """
    model.to(device)
    best_val_acc = 0.0
    best_model_path = os.path.join(OUTPUT_DIR, "eaml_bestmodel.pth")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        # Training loop with progress bar
        train_progress = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
        for images, texts, labels in train_progress:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            #outputs = model(images, texts)
            #images, texts = images.to(device), {key: val.to(device) for key, val in texts.items()}  # Move both to the same device
            if isinstance(texts, tuple):  # Ensure texts is a dictionary
                texts = texts[0]  # Extract the dictionary from the tuple

            images = images.to(device)
            if isinstance(texts, str):  # 🚨 Ensure texts is tokenized before sending it to the model
                texts = tokenizer(texts, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
            texts = {key: val.to(device) for key, val in texts.items()}
            print(type(texts), texts)

            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            train_progress.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{(correct/total)*100:.2f}%"})

        # Calculate training accuracy
        train_loss = running_loss / len(loaders["train"])
        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

        # Validation
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at {best_model_path}")

    print("Training Completed.")
    return best_model_path

def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model on validation/test dataset.

    Args:
        model: The trained model.
        data_loader: DataLoader for validation or test set.
        criterion: Loss function.
        device: Device ('cuda' or 'cpu').

    Returns:
        loss: Average loss on the dataset.
        accuracy: Accuracy on the dataset.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        eval_progress = tqdm(data_loader, desc="[Evaluating]", leave=False)
        for images, texts, labels in eval_progress:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            eval_progress.set_postfix({"Loss": f"{loss.item():.4f}"})

    accuracy = 100 * correct / total
    return total_loss / len(data_loader), accuracy

def test(model, test_loader, criterion, device):
    """
    Test the trained model on the test dataset.

    Args:
        model: The trained model.
        test_loader: DataLoader for test dataset.
        criterion: Loss function.
        device: Device ('cuda' or 'cpu').
    
    Returns:
        test_accuracy: Accuracy on the test dataset.
    """
    print("Starting Model Testing...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_acc

def save_training_log(model_name, optimizer_name, epochs, batch_size, num_classes, learning_rate, test_accuracy):
    """
    Save training details to a CSV log file.

    Args:
        model_name: Name of the trained model.
        optimizer_name: Optimizer used.
        epochs: Number of epochs.
        batch_size: Batch size.
        num_classes: Number of classes.
        learning_rate: Learning rate.
        test_accuracy: Final test accuracy.
    """
    log_data = {
        "Model": model_name,
        "Optimizer": optimizer_name,
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Num Classes": num_classes,
        "Learning Rate": learning_rate,
        "Test Accuracy (%)": f"{test_accuracy:.2f}",
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    log_file = "logs/training_log.csv"
    os.makedirs("logs", exist_ok=True)

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_data])], ignore_index=True)
    else:
        df = pd.DataFrame([log_data])

    df.to_csv(log_file, index=False)
    print(f"Training log saved at {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAML Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device for training")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names to load (default: all classes)")

    args = parser.parse_args()

    num_classes = len(args.classes) if args.classes else 16
    print(f"Using classes: {args.classes} and # of classes: {num_classes}")

    # Load dataset
    data_loader = DataLoader(
        data_dir=args.data_dir, 
        batch_size=args.batch_size, 
        dataset_type="all", 
        classes=args.classes, 
        use_eaml=True
        )
    loaders = data_loader.load_data()

    # Initialize model
    model = EAMLModel(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()

    # Train model
    best_model_path = train(model, loaders, args.epochs, optimizer, criterion, args.device)

    # Load the best model for testing
    model.load_state_dict(torch.load(best_model_path))
    test_accuracy = test(model, loaders["test"], criterion, args.device)

    # Save training log
    save_training_log(args.model_name, "adam", args.epochs, args.batch_size, num_classes, args.learning_rate, test_accuracy)
