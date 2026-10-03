import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers import BertTokenizer
from PIL import Image
import pytesseract  # OCR for extracting text from images
import torch

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class EAML_Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.texts = []
        self.labels = []  # Stores class labels

        self.load_samples()

    def extract_text_from_image(self, img_path, txt_path):
        """
        Extract text from an image using Tesseract OCR and save it as a text file.
        """
        image = Image.open(img_path).convert("RGB")
        extracted_text = pytesseract.image_to_string(image).strip()

        if not extracted_text:
            extracted_text = "No Text"  # Handle cases where OCR fails

        # Save extracted text to a .txt file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)

        return extracted_text

    def load_samples(self):
        """
        Load image paths, extract text (if missing), and prepare dataset.
        """
        print(f"📂 Scanning dataset directory: {self.data_dir}")

        num_files = 0
        for root, _, files in os.walk(self.data_dir):
            for img_file in files:
                if img_file.endswith(".tif"):
                    img_path = os.path.join(root, img_file)
                    txt_path = img_path.replace(".tif", ".txt")
                    label = os.path.basename(root)  # Folder name as label

                    num_files += 1

                    # If text file does not exist, extract text using OCR
                    if not os.path.exists(txt_path):
                        print(f"⚠️ No text file found for {img_file}, extracting text...")
                        extracted_text = self.extract_text_from_image(img_path, txt_path)
                    else:
                        with open(txt_path, "r", encoding="utf-8") as file:
                            extracted_text = file.read().strip()

                    # Convert text to tokenized format
                    tokenized_text = tokenizer(extracted_text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

                    self.image_paths.append(img_path)
                    self.texts.append(tokenized_text)
                    self.labels.append(label)

        print(f"✅ Found {num_files} images, Loaded {len(self.image_paths)} valid samples from {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text = self.texts[idx]
        label = self.labels[idx]  # Returning label as well

        return image, text, label  # Ensures all three components are returned


class DataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224, dataset_type="all", classes=None, use_eaml=False):
        """
        Initializing the dataset loader.

        Parameters:
        data_dir: Path to the organized dataset directory.
        batch_size: Number of samples per batch.
        num_workers: Number of worker threads for data loading.
        img_size: Image size (for resizing).
        dataset_type: One of ['train', 'val', 'test', 'all'].
        classes: List of specific classes to load (default: all classes).
        use_eaml: If True, use the EAML dataset with image + text.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.classes = classes
        self.use_eaml = use_eaml  # Flag to determine dataset type

        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load datasets
        self.train_dataset = self.load_dataset("train")
        self.val_dataset = self.load_dataset("val")
        self.test_dataset = self.load_dataset("test")

    def eaml_load_dataset(self, dataset_type):
        """
        Custom function to load EAML datasets separately.
        """
        dataset_path = os.path.join(self.data_dir, dataset_type)

        if not os.path.exists(dataset_path):
            raise ValueError(f"❌ Error: EAML dataset directory does not exist: {dataset_path}")

        dataset = EAML_Dataset(dataset_path, transform=self.transform)

        print(f"✅ Loaded EAML dataset, Found {len(dataset)} samples.")

        return dataset

    def load_dataset(self, dataset_type):
        dataset_path = os.path.join(self.data_dir, dataset_type)

        if not os.path.exists(dataset_path):
            raise ValueError(f"❌ Error: {dataset_type} directory does not exist: {dataset_path}")

        if self.use_eaml:
            dataset = self.eaml_load_dataset(dataset_type)
        else:
            dataset = ImageFolder(root=dataset_path, transform=self.transform)

        print(f"✅ Loaded dataset: {dataset_type}, Found {len(dataset)} samples before filtering.")

        return dataset

    def get_data_loader(self, dataset_type):
        """Returns DataLoader object for the specified dataset type (train, val, test)."""
        dataset = self.train_dataset if dataset_type == "train" else self.val_dataset if dataset_type == "val" else self.test_dataset
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=(dataset_type == "train"), num_workers=self.num_workers)

    def load_data(self):
        """Loads train, val, test, or all data based on dataset_type."""
        loaders = {}
        if self.dataset_type in ["train", "all"]:
            loaders["train"] = self.get_data_loader("train")
        if self.dataset_type in ["val", "all"]:
            loaders["val"] = self.get_data_loader("val")
        if self.dataset_type in ["test", "all"]:
            loaders["test"] = self.get_data_loader("test")

        return loaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom DataLoader for Document Classification")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for resizing")
    parser.add_argument("--dataset_type", type=str, choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to load (train, val, test, or all)")
    parser.add_argument("--classes", nargs="+", default=None, help="List of class names to load (default: all classes)")
    parser.add_argument("--use_eaml", action="store_true", help="Enable EAML dataset (image + text pairs)")

    args = parser.parse_args()

    if isinstance(args.classes, str):
        args.classes = [c.strip() for c in args.classes.split(",")]

    loader = DataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        dataset_type=args.dataset_type,
        classes=args.classes,
        use_eaml=args.use_eaml
    )

    data_loaders = loader.load_data()
    print(f"✅ Data Loaders initialized for: {list(data_loaders.keys())}")
