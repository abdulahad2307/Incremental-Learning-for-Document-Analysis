import torch
import os

def load_checkpoint_info(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"\nLoaded checkpoint from: {checkpoint_path}")
    print(f"Available keys: {list(checkpoint.keys())}\n")

    # Print basic info
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    best_val_acc = checkpoint.get('best_val_acc', 'N/A')
    patience_counter = checkpoint.get('patience_counter', 'N/A')

    print(f"Epoch: {epoch}")
    print(f"Validation Loss: {val_loss}")
    print(f"Best Validation Accuracy: {best_val_acc}")
    print(f"Patience Counter: {patience_counter}")

    # Show model and optimizer state details
    if 'model_state_dict' in checkpoint:
        print(f"\nModel state dict: {len(checkpoint['model_state_dict'])} tensors loaded.")
    if 'optimizer_state_dict' in checkpoint:
        print(f"Optimizer state dict: {len(checkpoint['optimizer_state_dict']['state'])} parameter groups loaded.")
    if 'scheduler_state_dict' in checkpoint:
        print(f"Scheduler state dict: {len(checkpoint['scheduler_state_dict'])} entries loaded.")

    return checkpoint

if __name__ == "__main__":
    # Example usage:
    ckpt_path = "/home/woody/iwi5/iwi5280h/eaml_cil/eaml_cil_evm_training_KDILS_Final/best_model_specification.pth"
    load_checkpoint_info(ckpt_path)
