
# 基于深度学习的植物叶片病变检测系统

一个可运行的 **Python + PyTorch + Flask + ECharts** 植物叶片病害识别项目。项目支持：

- 自动从 Hugging Face 下载 PlantVillage 公开数据集图片
- 使用 PyTorch / torchvision 训练 MobileNetV2 图像分类模型
- Flask 后端提供图片上传与 URL 图片检测接口
- 前端使用 HTML/CSS/JavaScript + ECharts 展示病害概率分布
- 支持 VS Code / PyCharm / IDEA Ultimate 打开运行

> 说明：PlantVillage 数据集是公开植物叶片病害图像数据集，包含 54,306 张健康和病害叶片图片。官方 GitHub 仓库也推荐通过 Hugging Face `datasets` 加载数据。

---

## 1. 推荐开发工具

### 推荐：VS Code
本项目是 Python + Flask + 前端页面项目，用 VS Code 最合适。

推荐安装插件：

- Python
- Pylance
- Jupyter
- HTML CSS Support

### 也可以用 IDEA
如果你使用 IntelliJ IDEA Ultimate，可以安装 Python 插件后运行；普通 IDEA Community 更适合 Java 项目，不太推荐写这个 Python 项目。

---

## 2. 环境准备

建议 Python 版本：`3.10` 或 `3.11`

### Windows / macOS / Linux 通用命令

```bash
python -m venv .venv
```

Windows 激活：

```bash
.venv\Scripts\activate
```

macOS / Linux 激活：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果 PyTorch 安装比较慢，可以去 PyTorch 官网选择适合你电脑的命令安装。

---

## 3. 一键下载公开数据集图片

默认从 Hugging Face 下载 PlantVillage 的部分分类图片，保存为 PyTorch `ImageFolder` 目录结构。

```bash
python scripts/prepare_dataset_hf.py --max-per-class 120
```

默认下载 5 个类别：

- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy
- Tomato___Early_blight
- Tomato___healthy

生成目录：

```text
data/plantvillage/
├── train/
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── ...
└── val/
    ├── Potato___Early_blight/
    ├── Potato___Late_blight/
    └── ...
```

如果你想先查看所有可用类别：

```bash
python scripts/prepare_dataset_hf.py --list-classes
```

自定义类别示例：

```bash
python scripts/prepare_dataset_hf.py --max-per-class 100 --classes "Apple___Apple_scab" "Apple___healthy" "Tomato___Late_blight" "Tomato___healthy"
```

---

## 4. 训练模型

CPU 也能跑，只是速度慢一点。为了课程项目演示，训练 2~5 轮即可。

```bash
python train.py --data-dir data/plantvillage --epochs 3 --batch-size 16 --pretrained
```

训练完成后会生成：

```text
outputs/best_model.pth
outputs/class_names.json
```

---

## 5. 启动 Web 系统

```bash
python app.py
```

浏览器打开：

```text
http://127.0.0.1:5000
```

页面功能：

- 上传本地叶片图片检测
- 输入网络图片 URL 检测
- 展示预测类别、置信度、概率柱状图

---

## 6. 快速演示流程

完整命令：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/prepare_dataset_hf.py --max-per-class 120
python train.py --data-dir data/plantvillage --epochs 3 --batch-size 16 --pretrained
python app.py
```

---

## 7. 项目结构

```text
plant-disease-detection/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── plant_disease/
│   ├── __init__.py
│   ├── model.py
│   └── predict.py
├── scripts/
│   ├── prepare_dataset_hf.py
│   ├── prepare_dataset_from_urls.py
│   ├── quick_train_windows.bat
│   └── quick_train_linux_mac.sh
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/.gitkeep
├── data/
│   └── url_dataset_example.csv
├── outputs/.gitkeep
└── .vscode/
    └── launch.json
```

---

## 8. 简历写法

```text
基于深度学习的植物叶片病变检测系统｜个人深度学习项目
技术栈：Python、PyTorch、Flask、OpenCV、ECharts、HTML/CSS、JavaScript
开发工具：VS Code、Anaconda、Chrome DevTools、Postman、Git

项目描述：
本项目是一套面向农作物叶片健康检测的图像识别辅助系统。系统支持用户上传植物叶片图片或输入网络图片 URL，后端通过 PyTorch 训练好的图像分类模型对叶片病害类别进行识别，并将预测类别、置信度和各类别概率分布返回前端展示。

主要职责：
1. 使用 Flask 搭建后端服务，完成图片上传、URL 图片下载、图片格式校验、图像预处理和模型推理接口开发。
2. 使用 PyTorch 和 torchvision 基于 PlantVillage 公开数据集训练叶片病害分类模型，并保存模型权重用于 Web 推理。
3. 设计并实现前端上传页面和结果展示页，支持图片预览、检测提交、异常提示和结果渲染。
4. 使用 ECharts 对模型输出的各病害类别概率进行柱状图可视化展示，提高识别结果的可读性。
5. 针对图片较大导致上传和加载较慢的问题，增加前端图片预览压缩和后端文件大小限制，提升系统响应速度。
6. 使用 Postman 和浏览器开发者工具测试接口，覆盖正常图片、非图片文件、空文件、图片过大、URL 下载失败等异常场景。
```

---

## 9. 面试讲法

面试官问“你这个项目怎么做的？”可以这样回答：

> 这个项目主要是把深度学习图像分类模型接入到 Web 系统中。数据集方面我使用 PlantVillage 公开植物叶片病害数据集，先通过脚本从 Hugging Face 下载指定类别图片，然后用 PyTorch 的 ImageFolder 读取数据，基于 MobileNetV2 做迁移学习训练。Web 端使用 Flask 提供图片上传和 URL 图片检测接口，前端用 HTML/CSS/JavaScript 实现图片预览和提交，并用 ECharts 展示各病害类别的概率分布。这个项目的重点不是单纯训练模型，而是完成“数据下载、模型训练、模型推理、接口返回、前端可视化”的完整闭环。

---

## 10. 注意事项

1. 第一次下载数据集需要联网。
2. 第一次使用 `--pretrained` 训练时，torchvision 会下载 MobileNetV2 预训练权重。
3. 如果网络不好，可以先把 `--max-per-class` 调小，例如 30。
4. 如果没有 GPU，可以用 CPU 训练，课程演示建议只选 3~5 个类别。
5. 这个项目适合放简历，但面试时不要夸大成“高精度农业诊断系统”，应该强调是“课程/个人项目，用于学习 AI 模型 Web 部署”。
