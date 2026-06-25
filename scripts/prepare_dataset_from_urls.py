from __future__ import annotations

import argparse
import csv
import uuid
from pathlib import Path

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def download_image(url: str, target: Path, timeout: int = 15):
    headers = {"User-Agent": "Mozilla/5.0 plant-disease-demo"}
    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)
    try:
        with Image.open(target) as img:
            img.convert("RGB").save(target.with_suffix(".jpg"), quality=95)
        if target.suffix.lower() != ".jpg":
            target.unlink(missing_ok=True)
    except UnidentifiedImageError:
        target.unlink(missing_ok=True)
        raise ValueError("URL did not return a valid image")


def parse_args():
    parser = argparse.ArgumentParser(description="Download images from a CSV file for ImageFolder training")
    parser.add_argument("--csv", type=str, default="data/url_dataset_example.csv", help="CSV columns: split,label,url")
    parser.add_argument("--output-dir", type=str, default="data/url_dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows):
        split = row.get("split", "train").strip() or "train"
        label = row.get("label", "unknown").strip() or "unknown"
        url = row.get("url", "").strip()
        if not url:
            continue
        target = Path(args.output_dir) / split / label / f"{uuid.uuid4().hex}.tmp"
        try:
            download_image(url, target)
        except Exception as exc:
            print(f"下载失败：{url}，原因：{exc}")

    print("下载完成。训练命令示例：")
    print(f"python train.py --data-dir {args.output_dir} --epochs 3 --batch-size 16 --pretrained")


if __name__ == "__main__":
    main()
