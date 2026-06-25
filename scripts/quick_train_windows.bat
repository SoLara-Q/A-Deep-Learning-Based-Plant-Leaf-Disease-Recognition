@echo off
chcp 65001
python scripts\prepare_dataset_hf.py --max-per-class 120
python train.py --data-dir data\plantvillage --epochs 3 --batch-size 16 --pretrained
python app.py
pause
