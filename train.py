
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from plant_disease.model import build_model, device_name


def parse_args():
    parser = argparse.ArgumentParser(description="Train plant disease classifier")
    parser.add_argument("--data-dir", type=str, default="data/plantvillage", help="ImageFolder dataset directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows 建议保持 0")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained MobileNetV2 weights")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def build_dataloaders(data_dir: Path, img_size: int, batch_size: int, num_workers: int):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"没有找到数据集目录：{train_dir} 和 {val_dir}\n"
            "请先运行：python scripts/prepare_dataset_hf.py --max-per-class 120"
        )

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("训练集和验证集类别不一致，请重新下载数据集")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.classes


def run_one_epoch(model, loader, criterion, optimizer, device, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(device_name())
    print(f"使用设备：{device}")
    print(f"类别数量：{len(class_names)}")
    print("类别列表：")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")

    model = build_model(num_classes=len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    history = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, training=False)

        item = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        }
        history.append(item)
        print(
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "img_size": args.img_size,
                    "best_acc": best_acc,
                    "architecture": "mobilenet_v2",
                },
                output_dir / "best_model.pth",
            )
            print(f"已保存最佳模型：{output_dir / 'best_model.pth'}")

    with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    cost = time.time() - start
    print(f"\n训练完成，最佳验证准确率：{best_acc:.4f}，耗时：{cost:.1f}s")
    print("启动 Web：python app.py")


if __name__ == "__main__":
    main()
