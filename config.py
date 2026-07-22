"""Typed configuration and server environment overrides for DBSegNet."""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _pair(name: str, default: Tuple[int, int]) -> Tuple[int, int]:
    value = os.getenv(name)
    if not value:
        return default
    result = tuple(int(item.strip()) for item in value.split(","))
    if len(result) != 2:
        raise ValueError(f"{name} must contain two comma-separated integers")
    return result


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 2
    input_channels: int = 3
    branch: str = "dual"
    fusion_mode: str = "gated"
    use_cross_attention: bool = True
    initialization: str = "normal"
    checkpoint: str = ""


@dataclass(frozen=True)
class DataConfig:
    dataset_type: str = "voc"
    dataset_path: str = r"D:\code\DB_DATA\NRTS_combined_spatial_2023_label_corrected"
    split_dir: str = ""
    input_shape: Tuple[int, int] = (512, 512)
    augmentation: str = "planet_mild"
    no_augmentation_last_epochs: int = 20
    no_augmentation_mode: str = "resize_only"

    def resolved_split_dir(self) -> Path:
        if self.split_dir:
            return Path(self.split_dir)
        if self.dataset_type == "voc":
            return Path(self.dataset_path) / "VOC2007" / "ImageSets" / "Segmentation"
        raise ValueError("CAS requires split_dir (or SPLIT_DIR)")


@dataclass(frozen=True)
class TrainConfig:
    cuda: bool = True
    distributed: bool = False
    sync_bn: bool = False
    fp16: bool = True
    seed: int = 11
    epochs: int = 80
    batch_size: int = 4
    workers: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    momentum: float = 0.9
    lr_schedule: str = "cos"
    dice_loss: bool = True
    dice_weight: float = 1.0
    focal_loss: bool = False
    label_smoothing: float = 0.0
    eval_enabled: bool = True
    eval_period: int = 5
    save_period: int = 10
    save_dir: str = r"D:\code\DB_DATA\logs\NDBSegNet_reproduction"
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_start_epoch: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(defaults: Optional[ExperimentConfig] = None) -> ExperimentConfig:
    defaults = defaults or ExperimentConfig(ModelConfig(), DataConfig(), TrainConfig())
    m, d, t = defaults.model, defaults.data, defaults.train

    model = ModelConfig(
        num_classes=int(os.getenv("NUM_CLASSES", str(m.num_classes))),
        input_channels=int(os.getenv("INPUT_CHANNELS", str(m.input_channels))),
        branch=os.getenv("MODEL_BRANCH", m.branch).lower(),
        fusion_mode=os.getenv("FUSION_MODE", m.fusion_mode).lower(),
        use_cross_attention=not _boolean("FORCE_NO_CA", not m.use_cross_attention),
        initialization=os.getenv("INIT_TYPE", m.initialization).lower(),
        checkpoint=os.getenv("MODEL_PATH", m.checkpoint),
    )
    data = DataConfig(
        dataset_type=os.getenv("DATASET_TYPE", d.dataset_type).lower(),
        dataset_path=os.getenv(
            "DATASET_PATH", os.getenv("VOC_DATASET_PATH", d.dataset_path)
        ),
        split_dir=os.getenv("SPLIT_DIR", d.split_dir),
        input_shape=_pair("INPUT_SHAPE", d.input_shape),
        augmentation=os.getenv("AUG_MODE", d.augmentation).lower(),
        no_augmentation_last_epochs=int(
            os.getenv(
                "NO_AUG_LAST_EPOCHS", str(d.no_augmentation_last_epochs)
            )
        ),
        no_augmentation_mode=os.getenv(
            "NO_AUG_MODE", d.no_augmentation_mode
        ).lower(),
    )
    train = TrainConfig(
        cuda=_boolean("CUDA", t.cuda),
        distributed=_boolean("DISTRIBUTED", t.distributed),
        sync_bn=_boolean("SYNC_BN", t.sync_bn),
        fp16=_boolean("FP16", t.fp16),
        seed=int(os.getenv("SEED", str(t.seed))),
        epochs=int(os.getenv("EPOCHS", os.getenv("UNFREEZE_EPOCHS", str(t.epochs)))),
        batch_size=int(os.getenv("BATCH_SIZE", str(t.batch_size))),
        workers=int(os.getenv("NUM_WORKERS", str(t.workers))),
        optimizer=os.getenv("OPTIMIZER", t.optimizer).lower(),
        learning_rate=float(os.getenv("LEARNING_RATE", str(t.learning_rate))),
        min_learning_rate=float(
            os.getenv("MIN_LEARNING_RATE", str(t.min_learning_rate))
        ),
        weight_decay=float(os.getenv("WEIGHT_DECAY", str(t.weight_decay))),
        momentum=float(os.getenv("MOMENTUM", str(t.momentum))),
        lr_schedule=os.getenv("LR_SCHEDULE", t.lr_schedule).lower(),
        dice_loss=_boolean("DICE_LOSS", t.dice_loss),
        dice_weight=float(os.getenv("DICE_WEIGHT", str(t.dice_weight))),
        focal_loss=_boolean("FOCAL_LOSS", t.focal_loss),
        label_smoothing=float(
            os.getenv("LABEL_SMOOTHING", str(t.label_smoothing))
        ),
        eval_enabled=_boolean("EVAL_ENABLED", t.eval_enabled),
        eval_period=int(os.getenv("EVAL_PERIOD", str(t.eval_period))),
        save_period=int(os.getenv("SAVE_PERIOD", str(t.save_period))),
        save_dir=os.getenv("SAVE_DIR", t.save_dir),
        use_ema=_boolean("USE_EMA", t.use_ema),
        ema_decay=float(os.getenv("EMA_DECAY", str(t.ema_decay))),
        ema_start_epoch=int(
            os.getenv("EMA_START_EPOCH", str(t.ema_start_epoch))
        ),
    )
    config = ExperimentConfig(model, data, train)
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.data.dataset_type not in {"voc", "cas"}:
        raise ValueError("dataset_type must be 'voc' or 'cas'")
    if not config.data.dataset_path:
        raise ValueError("dataset_path must point to an external dataset")
    if config.model.branch not in {"dual", "cnn", "tr"}:
        raise ValueError("branch must be 'dual', 'cnn' or 'tr'")
    if config.model.fusion_mode not in {"gated", "sum"}:
        raise ValueError("fusion_mode must be 'gated' or 'sum'")
    if config.train.optimizer not in {"adam", "adamw", "sgd"}:
        raise ValueError("optimizer must be 'adam', 'adamw' or 'sgd'")
    if config.data.augmentation not in {
        "default", "weak", "planet_mild", "resize_only", "direct_resize"
    }:
        raise ValueError("unsupported augmentation mode")
