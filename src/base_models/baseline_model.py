import os
import datetime
import argparse
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from tqdm import tqdm  # For progress bars

from utils.baseline_dataloader import DataLoader

OUTPUT_DIR = "/home/woody/iwi5/iwi5280h/saved_models"

class BaselineModel:
    def __init__(self, model_name, num_classes, optimizer_name, learning_rate, device="cuda"):
        """
        Initializes the baseline model.

        Parameters:
        model_name: Name of the model ('resnet50' or 'densenet121').
        num_classes: Number of output classes.
        learning_rate: Learning rate for optimizer.
        device: Device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model_name = model_name
        # Load model from timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer_name = optimizer_name
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer. Choose 'adam', 'sgd', or 'adamw'.")

        # Learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        # Early stopping
        self.best_val_loss = float("inf")
        self.early_stop_patience = 10  # Stop if validation loss doesn't improve for 10 epochs
        self.early_stop_counter = 0

    def train(self, train_loader, val_loader, epochs=10):
        """
        Train the model.

        Parameters:
        train_loader: loads the training data
        val_loader: loads the validation data
        epochs: number of epochs
        """
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            # Training loop with progress bar
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False)
            for inputs, labels in train_progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                # Update progress bar description
                train_progress.set_postfix({"Training Loss": f"{loss.item():.4f}"})

            # Print training loss
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")

            # Validation loop with progress bar
            val_loss, val_metrics = self.evaluate(val_loader)
            print(
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_metrics['accuracy']:.2f}%, "
                f"Validation F1 Score: {val_metrics['f1_score']:.4f}, "
                f"Validation Precision: {val_metrics['precision']:.4f}, "
                f"Validation Recall: {val_metrics['recall']:.4f}, "
                f"Validation AUROC: {val_metrics['auroc']:.4f}"
            )

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0

                # Save the best model
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                model_save_path = os.path.join(OUTPUT_DIR, f"{self.model_name}_best_epoch_{epoch+1}.pth")
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Best model saved at {model_save_path}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} as validation loss did not improve for {self.early_stop_patience} epochs.")
                    break

    def evaluate(self, val_loader):
        """Evaluate the model.

        Parameters:
        val_loader: loads the validation data

        Returns:
        val_loss: Average validation loss
        metrics: Dictionary containing accuracy, F1 score, precision, recall, and AUROC
        """
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        # Validation loop with progress bar
        val_progress = tqdm(val_loader, desc="[Validation]", leave=False)
        with torch.no_grad():
            for inputs, labels in val_progress:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Get predicted class and probabilities
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                # Update progress bar description
                val_progress.set_postfix({"Validation Loss": f"{loss.item():.4f}"})

        val_loss /= len(val_loader)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        # Calculate AUROC (only for binary classification or one-vs-rest for multi-class)
        if self.num_classes == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(all_labels, all_probs, multi_class="ovo", average="weighted")

        metrics = {
            "accuracy": accuracy * 100,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "auroc": auroc,
        }

        return val_loss, metrics

    def test(self, test_loader):
        """
        Test the model.

        Parameters:
        test_loader: loads the test data

        Returns:
        test_metrics: Dictionary containing accuracy, F1 score, precision, recall, and AUROC
        """
        # Testing loop with progress bar
        test_progress = tqdm(test_loader, desc="[Testing]", leave=False)
        _, test_metrics = self.evaluate(test_loader)
        return test_metrics

    @staticmethod
    def save_training_log(model_name, optimizer_name, epochs, batch_size, num_classes, learning_rate, test_metrics):
        """
        Save training details to a CSV log file.
        """
        log_data = {
            "Model": model_name,
            "Optimizer": optimizer_name,
            "Epochs": epochs,
            "Batch Size": batch_size,
            "Num Classes": num_classes,
            "Learning Rate": learning_rate,
            "Test Accuracy (%)": f"{test_metrics['accuracy']:.2f}",
            "Test F1 Score": f"{test_metrics['f1_score']:.4f}",
            "Test Precision": f"{test_metrics['precision']:.4f}",
            "Test Recall": f"{test_metrics['recall']:.4f}",
            "Test AUROC": f"{test_metrics['auroc']:.4f}",
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        log_file = "logs/training_log.csv"
        os.makedirs("logs", exist_ok=True)

        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
            df = df.append(log_data, ignore_index=True)
        else:
            df = pd.DataFrame([log_data])

        df.to_csv(log_file, index=False)
        print(f"Training log saved at {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Model for Document Classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to organized dataset directory")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121"], default="resnet50",
                        help="Model to use (resnet50 or densenet121)")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names to load (default: all classes)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd", "adamw"], default="adam", help="Optimizer to use")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Where the best model is saved")

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir

    num_classes = len(args.classes) if args.classes else 16

    print(f"Using classes: {args.classes} and # of classes: {num_classes}") 

    ## Loading the dataset
    data_loader = DataLoader(data_dir=args.data_dir, batch_size=args.batch_size, dataset_type="all", classes=args.classes)
    loaders = data_loader.load_data()

    ## Initializing and training model
    model = BaselineModel(model_name=args.model_name, num_classes=num_classes, optimizer_name=args.optimizer, learning_rate=args.learning_rate, device=args.device)
    model.train(loaders["train"], loaders["val"], epochs=args.epochs)
    test_metrics = model.test(loaders["test"])
    print(
        f"Test Accuracy: {test_metrics['accuracy']:.2f}%, "
        f"Test F1 Score: {test_metrics['f1_score']:.4f}, "
        f"Test Precision: {test_metrics['precision']:.4f}, "
        f"Test Recall: {test_metrics['recall']:.4f}, "
        f"Test AUROC: {test_metrics['auroc']:.4f}"
    )

    model.save_training_log(args.model_name, args.optimizer, args.epochs, args.batch_size, num_classes, 
                      args.learning_rate, test_metrics)