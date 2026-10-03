import torch

class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model parameters
    EMBED_DIM = 512
    NUM_HEADS = 8
    NUM_CLASSES = 16  # Will be updated based on actual classes
    
    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    BETA = 0.5  # For Tr-KLD loss
    
    # Paths
    OUTPUT_DIR = 'outputs'
    LOG_DIR = 'logs'
    
    # OCR
    OCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
    
config = Config()