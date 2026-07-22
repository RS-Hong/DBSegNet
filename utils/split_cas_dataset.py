import json
import os
import random


DATASET_PATH = r"D:\code\DB_DATA\数据集"
SPLIT_DIR = r"D:\code\DBSegNet\splits\cas_seed11_70_30"
SEED = 11
TRAIN_RATIO = 0.7
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def list_files(path):
    return sorted(
        f for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(IMAGE_EXTS)
    )


def main():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    train_lines = []
    val_lines = []
    summary = {
        "dataset_path": DATASET_PATH,
        "split_dir": SPLIT_DIR,
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "subdatasets": [],
    }

    for subdataset in sorted(os.listdir(DATASET_PATH)):
        sub_path = os.path.join(DATASET_PATH, subdataset)
        img_dir = os.path.join(sub_path, "img")
        mask_dir = os.path.join(sub_path, "mask")
        if not os.path.isdir(sub_path) or not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        image_names = list_files(img_dir)
        mask_names = list_files(mask_dir)
        image_lower = {name.lower(): name for name in image_names}
        mask_lower = {name.lower(): name for name in mask_names}
        paired_keys = sorted(set(image_lower).intersection(mask_lower))
        paired = [image_lower[key] for key in paired_keys]
        missing_masks = [image_lower[key] for key in sorted(set(image_lower).difference(mask_lower))]
        missing_images = [mask_lower[key] for key in sorted(set(mask_lower).difference(image_lower))]

        rng.shuffle(paired)
        train_count = int(len(paired) * TRAIN_RATIO)
        train_items = sorted(paired[:train_count])
        val_items = sorted(paired[train_count:])

        train_lines.extend(f"{subdataset}/{name}\n" for name in train_items)
        val_lines.extend(f"{subdataset}/{name}\n" for name in val_items)

        summary["subdatasets"].append({
            "name": subdataset,
            "paired": len(paired),
            "train": len(train_items),
            "val": len(val_items),
            "missing_masks": missing_masks,
            "missing_images": missing_images,
        })

    with open(os.path.join(SPLIT_DIR, "train.txt"), "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    with open(os.path.join(SPLIT_DIR, "val.txt"), "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    with open(os.path.join(SPLIT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("CAS split saved to:", SPLIT_DIR)
    print("train:", len(train_lines), "val:", len(val_lines))


if __name__ == "__main__":
    main()
