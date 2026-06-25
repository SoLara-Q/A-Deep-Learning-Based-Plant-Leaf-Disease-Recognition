import argparse
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

TARGET_CLASSES = {
    "Tomato___healthy",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Apple___healthy",
}


def safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]', "_", str(name))


def extract_path(item):
    for value in item.values():
        if isinstance(value, str):
            m = re.search(r'raw[/\\]color[/\\][^\n\r]+?\.(JPG|jpg|JPEG|jpeg|png|PNG)', value)
            if m:
                return m.group(0).replace("\\", "/")
    return None


def parse_class_name(path: str):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 4:
        return parts[2]
    return None


def download_image(rel_path: str, save_path: Path):
    url = GITHUB_RAW_PREFIX + quote(rel_path, safe="/")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                time.sleep(1)
                continue

            tmp = save_path.with_suffix(".tmp")
            tmp.write_bytes(r.content)

            img = Image.open(tmp).convert("RGB")
            img.save(save_path, format="JPEG", quality=90)
            tmp.unlink(missing_ok=True)
            return True
        except Exception:
            time.sleep(1)

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="data/plantvillage")
    parser.add_argument("--max-per-class", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"删除旧数据目录：{output_dir}")
        shutil.rmtree(output_dir)

    print("正在读取 PlantVillage 图片路径清单...")
    dataset = load_dataset(DATASET_NAME, "default", split="train")

    paths = []
    for item in dataset:
        rel_path = extract_path(item)
        if not rel_path:
            continue

        class_name = parse_class_name(rel_path)
        if class_name in TARGET_CLASSES:
            paths.append((rel_path, safe_name(class_name)))

    random.shuffle(paths)

    selected = defaultdict(int)
    saved = defaultdict(lambda: {"train": 0, "val": 0})

    target_total = len(TARGET_CLASSES) * args.max_per_class
    print(f"目标类别数：{len(TARGET_CLASSES)}")
    print(f"目标下载图片数：{target_total}")

    for rel_path, class_name in paths:
        if selected[class_name] >= args.max_per_class:
            continue

        current = selected[class_name]
        val_count = max(1, int(args.max_per_class * args.val_ratio))
        subset = "val" if current < val_count else "train"

        save_path = output_dir / subset / class_name / f"{current:05d}.jpg"

        if download_image(rel_path, save_path):
            selected[class_name] += 1
            saved[class_name][subset] += 1
            total = sum(v["train"] + v["val"] for v in saved.values())
            print(f"已下载 {total}/{target_total}：{class_name}")

        if all(selected[safe_name(c)] >= args.max_per_class for c in TARGET_CLASSES):
            break

    print("\n数据集准备完成！")
    for class_name, count in saved.items():
        print(f"{class_name}: train={count['train']}, val={count['val']}")


if __name__ == "__main__":
    main()
