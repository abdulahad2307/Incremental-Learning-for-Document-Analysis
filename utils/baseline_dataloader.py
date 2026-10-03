"""Image-only loaders for the CNN baseline (src/base_models/baseline_model.py): ImageFolder per split, optional class subset."""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision.datasets import ImageFolder

class DataLoader:
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224, dataset_type="all", classes=None):
        """
        Initializing the dataset loader.

        Parameters:
        data_dir: Path to the organized dataset directory.
        batch_size: Number of samples per batch.
        num_workers: Number of worker threads for data loading.
        img_size: Image size (for resizing).
        dataset_type: One of ['train', 'val', 'test', 'all'].
        classes: List of specific classes to load (default: all classes).
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.classes = classes

        # Defining image transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load datasets
        self.train_dataset = self.load_dataset("train")
        self.val_dataset = self.load_dataset("val")
        self.test_dataset = self.load_dataset("test")

    def load_dataset(self, dataset_type):
        dataset_path = os.path.join(self.data_dir, dataset_type)

        if not os.path.exists(dataset_path):
            raise ValueError(f"{dataset_type} directory does not exist: {dataset_path}")

        dataset = ImageFolder(root=dataset_path, transform=self.transform)

        print(f"Loaded dataset: {dataset_type}, Found {len(dataset.samples)} samples before filtering.")

        # If specific classes are provided, filter the dataset
        if self.classes is not None:
            # Create a mapping from class names to indices
            class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
            
            # Filter samples and targets
            filtered_samples = []
            filtered_targets = []
            for sample, target in dataset.samples:
                class_name = dataset.classes[target]
                if class_name in self.classes:
                    filtered_samples.append((sample, class_to_idx[class_name]))
                    filtered_targets.append(class_to_idx[class_name])

            dataset.samples = filtered_samples
            dataset.targets = filtered_targets
            dataset.class_to_idx = class_to_idx
            dataset.classes = self.classes

            print(f"Dataset after filtering by classes: {dataset_type}, Remaining samples: {len(dataset.samples)}")

            if len(dataset.samples) == 0:
                raise ValueError(f"Filtered dataset is empty! Check if class names are correct: {self.classes}")

        return dataset
    
    def get_data_loader(self, dataset_type):
        """
        This function returns a DataLoader object for the specified dataset type (train, val, test).
        """
        if dataset_type == "train":
            dataset = self.train_dataset
        elif dataset_type == "val":
            dataset = self.val_dataset
        elif dataset_type == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")

        # Ensures correct DataLoader initialization with shuffle only for training
        return TorchDataLoader(dataset, batch_size=self.batch_size, shuffle=(dataset_type == "train"), num_workers=self.num_workers)


    def load_data(self):
        """
        Loads train, val, test, or all data based on dataset_type.
        """
        loaders = {}

        if self.dataset_type in ["train", "all"]:
            loaders["train"] = self.get_data_loader("train")
        if self.dataset_type in ["val", "all"]:
            loaders["val"] = self.get_data_loader("val")
        if self.dataset_type in ["test", "all"]:
            loaders["test"] = self.get_data_loader("test")

        return loaders
