from src.class_incremental.eaml.class_incremental import run_incremental_learning

CLASS_ORDER = ["letter", "form", "email"]  # Just 3 classes

run_incremental_learning(
    data_root="/home/woody/iwi5/iwi5280h/dataset/small_dataset",
    class_order=CLASS_ORDER,
    base_model_path="checkpoints/pretrained/docformer.pth",
    model_name="docformer",
    checkpoint_dir="checkpoints/cil_test",
    start_step=0,
    batch_size=8,
    lr=2e-5,
    num_epochs=5
)
