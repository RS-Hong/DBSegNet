"""Reproduce the final DBSegNet training protocol."""

from config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, load_config
from training.runner import run_training


# Change only this switch and the three paths for a different machine.
RUN_ON_SERVER = True

if RUN_ON_SERVER:
    DATASET_PATH = "/root/autodl-tmp/NDBSegNet/NRTS_combined_spatial"
    SPLIT_DIR = "/root/autodl-tmp/NDBSegNet/splits/81_5"
    SAVE_DIR = "/root/autodl-tmp/NDBSegNet/logs/DBSegNet_81_5_reproduction"
else:
    DATASET_PATH = r"D:\code\DB_DATA\NRTS_combined_spatial_2023_label_corrected"
    SPLIT_DIR = r"D:\code\DB_DATA\DBSegNet_81_5_historical_split_reference"
    SAVE_DIR = r"D:\code\DB_DATA\logs\NDBSegNet_81_5_reproduction"


# Model used for the reported DBSegNet result.
MODEL = ModelConfig(
    num_classes=2,
    input_channels=3,
    branch="dual",
    fusion_mode="gated",
    use_cross_attention=True,
    initialization="normal",
    checkpoint="",
)

# Use the retained fused data with the archived 81.5% train/validation split.
DATA = DataConfig(
    dataset_type="voc",
    dataset_path=DATASET_PATH,
    split_dir=SPLIT_DIR,
    input_shape=(512, 512),
    augmentation="planet_mild",
    no_augmentation_last_epochs=20,
    no_augmentation_mode="resize_only",
)

# Training protocol used by the 81.5% experiment series.
TRAIN = TrainConfig(
    cuda=True,
    distributed=False,
    sync_bn=False,
    fp16=True,
    seed=11,
    epochs=100,
    batch_size=4,
    workers=4,
    optimizer="adamw",
    learning_rate=1e-4,
    min_learning_rate=1e-6,
    weight_decay=1e-2,
    momentum=0.9,
    lr_schedule="cos",
    dice_loss=True,
    dice_weight=1.0,
    focal_loss=False,
    label_smoothing=0.0,
    eval_enabled=True,
    eval_period=5,
    save_period=10,
    save_dir=SAVE_DIR,
    use_ema=False,
    ema_decay=0.999,
    ema_start_epoch=1,
)

DEFAULT_CONFIG = ExperimentConfig(MODEL, DATA, TRAIN)


if __name__ == "__main__":
    run_training(load_config(DEFAULT_CONFIG))
