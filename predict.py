"""Predict a single image with the final DBSegNet checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nets import DBSegNet
from training.checkpoint import load_matching_weights
from utils.utils import cvtColor, preprocess_input, resize_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--no-cross-attention", action="store_true")
    parser.add_argument("--fusion-mode", choices=("gated", "sum"), default="gated")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBSegNet(
        num_classes=2,
        fusion_mode=args.fusion_mode,
        use_cross_attention=not args.no_cross_attention,
    )
    report = load_matching_weights(model, args.checkpoint)
    model.to(device).eval()
    print(f"Loaded checkpoint: {report}")

    image = cvtColor(Image.open(args.input))
    original_width, original_height = image.size
    resized, width, height = resize_image(
        image, (args.input_size, args.input_size)
    )
    array = preprocess_input(np.asarray(resized, dtype=np.float32))
    tensor = torch.from_numpy(array.transpose(2, 0, 1)[None]).to(device)

    with torch.no_grad():
        logits = model(tensor)[0]
        probability = torch.softmax(logits, dim=0)[1].cpu().numpy()
    top = (args.input_size - height) // 2
    left = (args.input_size - width) // 2
    probability = probability[top : top + height, left : left + width]
    probability = Image.fromarray(probability.astype(np.float32), mode="F")
    probability = probability.resize(
        (original_width, original_height), Image.BILINEAR
    )
    mask = (np.asarray(probability) >= 0.5).astype(np.uint8) * 255

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output)
    print(output)


if __name__ == "__main__":
    main()

