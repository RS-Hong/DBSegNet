"""Checkpoint loading shared by training and inference."""

from pathlib import Path
from typing import Dict

import torch


def load_matching_weights(model: torch.nn.Module, checkpoint_path: str) -> Dict[str, int]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")

    source = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    target.update(matched)
    model.load_state_dict(target)
    return {
        "matched": len(matched),
        "checkpoint_tensors": len(source),
        "model_tensors": len(target),
    }

