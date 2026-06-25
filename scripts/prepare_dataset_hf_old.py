from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DEFAULT_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
]


def safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text)


def get_label_names(config: str):
    # Load a tiny non-streaming object to read features. If this fails in future datasets versions,
    # the code below still tries to read labels while streaming.
    dataset = load_dataset("mohanty/PlantVillage", config, split="train[:1]")
    label_feature = dataset.features.get("label")
    if hasattr(label_feature, "names"):
        return list(label_feature.names)
    raise RuntimeError("无法读取数据集标签，请检查 datasets 版本")


def print_classes(config: str):
    labels = get_label_names(config)
    print("PlantVillage 可用类别：")
    for item in labels:
        print(item)


def save_pil_image(image: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, quality=95)


def prepare_split(split: str, target_split: str, args, label_names, selected_classes):
    out_root = Path(args.output_dir) / target_split
    max_per_class = args.max_per_class if target_split == "train" else max(10, args.max_per_class // 4)
    counts = defaultdict(int)
    total_needed = max_per_class * len(selected_classes)

    print(f"\n开始下载 {split} -> {target_split}，每类最多 {max_per_class} 张")
    ds = load_dataset("mohanty/PlantVillage", args.config, split=split, streaming=True)

    progress = tqdm(total=total_needed)
    for idx, item in enumerate(ds):
        raw_label = item.get("label")
        if isinstance(raw_label, int):
            label = label_names[raw_label]
        else:
            label = str(raw_label)

        if label not in selected_classes:
            continue
        if counts[label] >= max_per_class:
            continue

        image = item.get("image")
        if image is None:
            continue

        file_name = f"{safe_filename(label)}_{counts[label]:05d}.jpg"
        save_path = out_root / label / file_name
        save_pil_image(image, save_path)
        counts[label] += 1
        progress.update(1)

        if all(counts[c] >= max_per_class for c in selected_classes):
            break
    progress.close()

    print(f"{target_split} 下载结果：")
    for c in selected_classes:
        print(f"  {c}: {counts[c]} 张")

    missing = [c for c in selected_classes if counts[c] == 0]
    if missing:
        print("\n警告：以下类别没有下载到图片，请运行 --list-classes 检查类别名称是否正确：")
        for c in missing:
            print("  -", c)


def parse_args():
    parser = argparse.ArgumentParser(description="Download PlantVillage images from Hugging Face")
    parser.add_argument("--output-dir", type=str, default="data/plantvillage")
    parser.add_argument("--config", type=str, default="color", choices=["color", "grayscale", "segmented"])
    parser.add_argument("--max-per-class", type=int, default=120, help="训练集每个类别最多下载多少张")
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--list-classes", action="store_true", help="只打印类别名称，不下载")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_classes:
        print_classes(args.config)
        return

    label_names = get_label_names(args.config)
    selected_classes = list(args.classes)

    unknown = [c for c in selected_classes if c not in label_names]
    if unknown:
        print("以下类别不存在，请检查拼写：")
        for c in unknown:
            print("  -", c)
        print("\n你可以运行下面命令查看所有类别：")
        print("python scripts/prepare_dataset_hf.py --list-classes")
        raise SystemExit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    prepare_split("train", "train", args, label_names, selected_classes)
    prepare_split("test", "val", args, label_names, selected_classes)
    print("\n数据集准备完成。下一步运行：")
    print("python train.py --data-dir data/plantvillage --epochs 3 --batch-size 16 --pretrained")


if __name__ == "__main__":
    main()
