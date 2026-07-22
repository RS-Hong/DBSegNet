"""Experiment naming, split fingerprints and configuration records."""

import datetime as dt
import hashlib
import json
from pathlib import Path


HISTORICAL_81_5_SPLITS = {
    "train": {
        "samples": 5719,
        "sha256": "6608D0F8DFDF50F022FCA21CF01A42C217E5799641716E55E92463C8BDF3FE8D",
    },
    "val": {
        "samples": 1450,
        "sha256": "B1B624E4BD2F16DCFCA1CA5CCCBEB9E11D55BDB0EB2B967CB85A817DC99E707F",
    },
}


def read_split(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line for line in path.read_text(encoding="utf-8-sig").splitlines(True) if line.strip()]


def split_fingerprint(path: Path):
    content = path.read_bytes()
    return {
        "path": str(path),
        "samples": len([line for line in content.decode("utf-8-sig").splitlines() if line.strip()]),
        "sha256": hashlib.sha256(content).hexdigest().upper(),
    }


def make_run_dir(config):
    timestamp = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    attention = "dff" if config.model.use_cross_attention else "lgf"
    loss = "ce_dice" if config.train.dice_loss else "ce"
    name = (
        f"{config.data.dataset_type}_DBSegNet_{config.model.branch}_"
        f"{config.model.fusion_mode}_{attention}_{config.data.augmentation}_"
        f"{loss}_ch{config.model.input_channels}_"
        f"noaug{config.data.no_augmentation_last_epochs}_"
        f"seed{config.train.seed}_{timestamp}"
    )
    run_dir = Path(config.train.save_dir) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run_manifest(config, run_dir, split_dir, effective_lr, final_lr):
    current = {
        name: split_fingerprint(split_dir / f"{name}.txt")
        for name in ("train", "val")
    }
    payload = config.as_dict()
    payload["resolved"] = {
        "run_dir": str(run_dir),
        "effective_initial_lr": effective_lr,
        "effective_final_lr": final_lr,
        "current_splits": current,
        "historical_81_5_splits": HISTORICAL_81_5_SPLITS,
        "same_as_historical_81_5": all(
            current[name]["samples"] == HISTORICAL_81_5_SPLITS[name]["samples"]
            and current[name]["sha256"] == HISTORICAL_81_5_SPLITS[name]["sha256"]
            for name in ("train", "val")
        ),
    }
    (run_dir / "train_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
