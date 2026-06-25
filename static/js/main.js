const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const urlBtn = document.getElementById('urlBtn');
const urlInput = document.getElementById('urlInput');
const previewImage = document.getElementById('previewImage');
const previewText = document.getElementById('previewText');
const uploadBox = document.querySelector('.upload-box');
const message = document.getElementById('message');
const resultPanel = document.getElementById('resultPanel');
const predictedLabel = document.getElementById('predictedLabel');
const confidence = document.getElementById('confidence');
const chartEl = document.getElementById('chart');
let chart = null;
let selectedFile = null;

function setMessage(text, isError = false) {
    message.textContent = text;
    message.classList.toggle('error', isError);
}

function showPreview(src) {
    previewImage.src = src;
    previewImage.style.display = 'block';
    previewText.style.display = 'none';
}

function renderResult(data) {
    const result = data.result;
    resultPanel.classList.remove('hidden');
    predictedLabel.textContent = result.predicted_display_label;
    confidence.textContent = `置信度：${result.confidence_percent}%`;

    if (data.image_url) {
        showPreview(data.image_url);
    }

    const chartData = result.all_probabilities
        .slice(0, 8)
        .map(item => ({ name: item.display_label, value: item.percent }));

    if (!chart) {
        chart = echarts.init(chartEl);
    }

    chart.setOption({
        tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
        grid: { left: 120, right: 24, top: 20, bottom: 20 },
        xAxis: { type: 'value', max: 100, axisLabel: { formatter: '{value}%' } },
        yAxis: { type: 'category', data: chartData.map(item => item.name) },
        series: [{
            type: 'bar',
            data: chartData.map(item => item.value),
            label: { show: true, position: 'right', formatter: '{c}%' },
            itemStyle: { borderRadius: [0, 8, 8, 0] }
        }]
    });
}

async function uploadAndPredict() {
    if (!selectedFile) {
        setMessage('请先选择一张叶片图片。', true);
        return;
    }
    const formData = new FormData();
    formData.append('file', selectedFile);
    setMessage('正在检测，请稍候...');

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.message || '检测失败');
        }
        setMessage('检测完成。');
        renderResult(data);
    } catch (err) {
        setMessage(err.message, true);
    }
}

async function predictUrl() {
    const url = urlInput.value.trim();
    if (!url) {
        setMessage('请输入图片 URL。', true);
        return;
    }
    setMessage('正在下载并检测网络图片...');

    try {
        const response = await fetch('/predict_url', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.message || '检测失败');
        }
        setMessage('检测完成。');
        renderResult(data);
    } catch (err) {
        setMessage(err.message, true);
    }
}

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = event => showPreview(event.target.result);
    reader.readAsDataURL(file);
    setMessage(`已选择图片：${file.name}`);
}

fileInput.addEventListener('change', event => {
    const file = event.target.files[0];
    if (file) handleFile(file);
});

uploadBtn.addEventListener('click', uploadAndPredict);
urlBtn.addEventListener('click', predictUrl);

uploadBox.addEventListener('dragover', event => {
    event.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', event => {
    event.preventDefault();
    uploadBox.classList.remove('dragover');
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
});

window.addEventListener('resize', () => {
    if (chart) chart.resize();
});
