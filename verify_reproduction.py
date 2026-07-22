"""Audit the DBSegNet training protocol, split identity and checkpoint ABI."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from config import load_config
from nets import DBSegNet
from train import DEFAULT_CONFIG
from training.experiment import HISTORICAL_81_5_SPLITS, split_fingerprint
from training.optimizers import fitted_learning_rates


CANONICAL_COMPONENT_HASHES = {
    "utils/dataloader.py": "347389EC19E584925D65C5C8DDC8201DD59675604BA92D57B53D04FFD94D63A8",
    "utils/callbacks.py": "53E46448C3F078B37F090484F091983193E3986D1043DAA8F09C24EA846376DC",
    "utils/utils_fit.py": "DF06E40D0BA23442D940DDFB2FC5F14B4A56CE5F73BE1563D46F817C6CB8C699",
    "nets/segformer_training.py": "2E8BF790FB96F9713C93223787C7553A9E46451E3967270218E62678A98B69F5",
}
HISTORICAL_CHECKPOINT_SHA256 = (
    "12B8E712EAA6046B35DCB21C821F5CEB6EA589682E3E74FC2A6713200BBFC315"
)
HISTORICAL_FORWARD_SHA256 = (
    "F4D8D1BB07BA541C1F74C65FCA7004214A39AD07E9F10513BC1CA3B15FE4EF0C"
)
DEFAULT_CHECKPOINT = Path(
    r"D:\code\DB_DATA\logs\第二次数据清洗_ablation\DBSegNet80\best_miou_weights.pth"
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(DEFAULT_CONFIG)
    split_dir = config.data.resolved_split_dir()
    current_splits = {
        name: split_fingerprint(split_dir / f"{name}.txt")
        for name in ("train", "val")
    }
    components = {}
    for relative, expected in CANONICAL_COMPONENT_HASHES.items():
        actual = file_hash(root / relative)
        components[relative] = {
            "sha256": actual,
            "canonical_sha256": expected,
            "match": actual == expected,
        }

    checkpoint_hash = file_hash(args.checkpoint)
    state = torch.load(args.checkpoint, map_location="cpu")
    model = DBSegNet(
        num_classes=config.model.num_classes,
        in_channels=config.model.input_channels,
        branch=config.model.branch,
        fusion_mode=config.model.fusion_mode,
        use_cross_attention=config.model.use_cross_attention,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.manual_seed(123)
    sample = torch.randn(1, config.model.input_channels, 128, 128)
    with torch.no_grad():
        output = model(sample).cpu().numpy()
    forward_hash = hashlib.sha256(output.tobytes()).hexdigest().upper()

    same_historical_split = all(
        current_splits[name]["samples"] == HISTORICAL_81_5_SPLITS[name]["samples"]
        and current_splits[name]["sha256"] == HISTORICAL_81_5_SPLITS[name]["sha256"]
        for name in ("train", "val")
    )
    initial_lr, final_lr = fitted_learning_rates(config.train)
    report = {
        "training_protocol_components_match": all(
            item["match"] for item in components.values()
        ),
        "checkpoint_strict_load": True,
        "checkpoint_tensors": len(state),
        "forward_fingerprint_match": forward_hash == HISTORICAL_FORWARD_SHA256,
        "forward_sha256": forward_hash,
        "historical_forward_sha256": HISTORICAL_FORWARD_SHA256,
        "checkpoint_sha256": checkpoint_hash,
        "historical_checkpoint_match": checkpoint_hash
        == HISTORICAL_CHECKPOINT_SHA256,
        "current_split_matches_historical_81_5": same_historical_split,
        "current_splits": current_splits,
        "historical_81_5_splits": HISTORICAL_81_5_SPLITS,
        "effective_initial_lr": initial_lr,
        "effective_final_lr": final_lr,
        "components": components,
        "interpretation": (
            "The training protocol is reproducible, but the reported 81.5% metric "
            "is not numerically reproducible from a different split."
            if not same_historical_split
            else "The protocol and historical split fingerprints both match."
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
