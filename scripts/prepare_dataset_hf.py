import argparse
import json
import random
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import load_dataset
from PIL import Image


DATASET_NAME = "mohanty/PlantVillage"
GITHUB_RAW_PREFIX = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/"


def safe_name(name: str) -> str:
    name = str(name)
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.replace(" ", "_")


def extract_path(item):
    for value in item.values():
        if isinstance(value, str):
            m = re.search(r'raw[/\\](color|grayscale|segmented)[/\\][^\n\r]+?\.(JPG|jpg|JPEG|jpeg|png|PNG)', value)
            if m:
                return m.group(0).replace("\\", "/")
    return None


def parse_class_name(path: str):
    # raw/color/Apple___Apple_scab/xxx.JPG
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 4:
        return parts[2]
    return None


def download_image(rel_path: str, save_path: Path, retry=3):
    url = GITHUB_RAW_PREFIX + quote(rel_path, safe="/")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for i in range(retry):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                time.sleep(1)
                continue

            tmp_path = save_path.with_suffix(".tmp")
            tmp_path.write_bytes(r.content)

            img = Image.open(tmp_path).convert("RGB")
            img.save(save_path, format="JPEG", quality=90)
            tmp_path.unlink(missing_ok=True)
            return True
        except Exception:
            time.sleep(1)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/plantvillage")
    parser.add_argument("--max-per-class", type=int, default=50)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-type", type=str, default="color", choices=["color", "grayscale", "segmented"])
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        print(f"发现旧数据目录：{output_dir}，正在删除并重新生成...")
        shutil.rmtree(output_dir)

    print("正在读取 Hugging Face 的 PlantVillage 图片路径清单...")
    dataset = load_dataset(DATASET_NAME, "default", split="train")

    print(f"清单总量：{len(dataset)}")
    print(f"字段：{dataset.column_names}")

    all_paths = []
    for item in dataset:
        rel_path = extract_path(item)
        if not rel_path:
            continue

        if f"raw/{args.image_type}/" not in rel_path:
            continue

        class_name = parse_class_name(rel_path)
        if not class_name:
            continue

        all_paths.append((rel_path, safe_name(class_name)))

    random.shuffle(all_paths)

    print(f"可用 {args.image_type} 图片路径数量：{len(all_paths)}")

    selected_count = defaultdict(int)
    saved_count = defaultdict(lambda: {"train": 0, "val": 0})
    class_to_index = {}

    for rel_path, class_name in all_paths:
        if selected_count[class_name] >= args.max_per_class:
            continue

        if class_name not in class_to_index:
            class_to_index[class_name] = len(class_to_index)

        current_index = selected_count[class_name]
        val_count = max(1, int(args.max_per_class * args.val_ratio))
        subset = "val" if current_index < val_count else "train"

        filename = f"{current_index:05d}.jpg"
        save_path = output_dir / subset / class_name / filename

        ok = download_image(rel_path, save_path)
        if ok:
            selected_count[class_name] += 1
            saved_count[class_name][subset] += 1
            total_saved = sum(v["train"] + v["val"] for v in saved_count.values())
            if total_saved % 20 == 0:
                print(f"已下载 {total_saved} 张图片...")
        else:
            print(f"下载失败，已跳过：{rel_path}")

    if not saved_count:
        raise RuntimeError("没有成功下载任何图片，请检查网络或 GitHub Raw 是否可访问。")

    mapping = {str(i): name for name, i in class_to_index.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print("\n数据集准备完成！")
    print(f"输出目录：{output_dir}")
    print(f"类别数量：{len(saved_count)}")

    for class_name, counts in saved_count.items():
        print(f"{class_name}: train={counts['train']}, val={counts['val']}")

    print("\n接下来运行：")
    print("python train.py --data-dir data/plantvillage --epochs 2 --batch-size 8 --pretrained")


if __name__ == "__main__":
    main()
