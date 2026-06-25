
from __future__ import annotations

import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from plant_disease.predict import Predictor

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "outputs" / "best_model.pth"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES

_predictor: Predictor | None = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "没有找到 outputs/best_model.pth。请先运行训练命令："
                "python train.py --data-dir data/plantvillage --epochs 3 --batch-size 16 --pretrained"
            )
        _predictor = Predictor(MODEL_PATH)
    return _predictor


def save_image_from_upload(file_storage) -> Path:
    filename = secure_filename(file_storage.filename or "upload.jpg")
    if not allowed_file(filename):
        raise ValueError("仅支持 png、jpg、jpeg、webp 图片格式")

    suffix = filename.rsplit(".", 1)[1].lower()
    target = UPLOAD_DIR / f"{uuid.uuid4().hex}.{suffix}"
    file_storage.save(target)

    try:
        with Image.open(target) as img:
            img.verify()
    except UnidentifiedImageError:
        target.unlink(missing_ok=True)
        raise ValueError("上传文件不是有效图片")

    return target


def save_image_from_url(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("图片链接必须以 http:// 或 https:// 开头")

    headers = {"User-Agent": "Mozilla/5.0 plant-disease-demo"}
    resp = requests.get(url, headers=headers, timeout=15, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()
    if "image" not in content_type:
        raise ValueError("该 URL 返回的内容不是图片")

    suffix = "jpg"
    if "png" in content_type:
        suffix = "png"
    elif "webp" in content_type:
        suffix = "webp"

    target = UPLOAD_DIR / f"{uuid.uuid4().hex}.{suffix}"
    total = 0
    with open(target, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                target.unlink(missing_ok=True)
                raise ValueError("网络图片过大，最大支持 8MB")
            f.write(chunk)

    try:
        with Image.open(target) as img:
            img.verify()
    except UnidentifiedImageError:
        target.unlink(missing_ok=True)
        raise ValueError("下载到的内容不是有效图片")

    return target


def build_response(image_path: Path) -> dict:
    predictor = get_predictor()
    result = predictor.predict(image_path)
    return {
        "success": True,
        "image_url": f"/static/uploads/{image_path.name}",
        "result": result,
    }


@app.route("/")
def index():
    return render_template("index.html", model_exists=MODEL_PATH.exists())


@app.route("/api/status")
def status():
    return jsonify({
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "message": "模型已就绪" if MODEL_PATH.exists() else "未找到模型，请先训练",
    })


@app.route("/predict", methods=["POST"])
def predict_upload():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "没有收到图片文件"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "请选择图片"}), 400
        image_path = save_image_from_upload(file)
        return jsonify(build_response(image_path))
    except FileNotFoundError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "message": f"检测失败：{exc}"}), 400


@app.route("/predict_url", methods=["POST"])
def predict_url():
    try:
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        if not url:
            return jsonify({"success": False, "message": "请输入图片 URL"}), 400
        image_path = save_image_from_url(url)
        return jsonify(build_response(image_path))
    except FileNotFoundError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "message": f"检测失败：{exc}"}), 400


@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=True)
