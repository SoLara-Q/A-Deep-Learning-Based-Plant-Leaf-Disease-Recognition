from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from plant_disease.model import build_model, device_name


def pretty_label(label: str) -> str:
    """Make PlantVillage labels easier to read on the web page."""
    return (
        label.replace("___", " - ")
        .replace("_", " ")
        .replace("(maize)", "maize")
        .replace("(including sour)", "including sour")
    )


class Predictor:
    def __init__(self, checkpoint_path: str | Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device_name())
        checkpoint: dict[str, Any] = torch.load(self.checkpoint_path, map_location=self.device)

        self.class_names = checkpoint["class_names"]
        self.img_size = int(checkpoint.get("img_size", 224))
        self.model = build_model(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, image_path: str | Path, topk: int = 5) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

        topk = min(topk, len(self.class_names))
        values, indices = torch.topk(probs, k=topk)

        all_probs = []
        for idx, score in enumerate(probs.tolist()):
            label = self.class_names[idx]
            all_probs.append({
                "label": label,
                "display_label": pretty_label(label),
                "probability": round(float(score), 6),
                "percent": round(float(score) * 100, 2),
            })
        all_probs.sort(key=lambda item: item["probability"], reverse=True)

        top_predictions = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            label = self.class_names[int(idx)]
            top_predictions.append({
                "label": label,
                "display_label": pretty_label(label),
                "probability": round(float(score), 6),
                "percent": round(float(score) * 100, 2),
            })

        best = top_predictions[0]
        return {
            "predicted_label": best["label"],
            "predicted_display_label": best["display_label"],
            "confidence": best["probability"],
            "confidence_percent": best["percent"],
            "top_predictions": top_predictions,
            "all_probabilities": all_probs,
        }
