#!/bin/bash

# 模型下载脚本

set -e

echo "=== 下载人脸识别模型 ==="

# 创建模型目录
mkdir -p models/insightface
mkdir -p models

# 下载InsightFace模型
echo "下载InsightFace模型..."

# 检查是否已安装git-lfs
if ! command -v git-lfs &> /dev/null; then
    echo "警告: 未找到git-lfs，某些模型可能无法下载"
fi

# 下载buffalo_l模型
echo "下载buffalo_l模型..."
if [ -d "insightface_models" ]; then
    rm -rf insightface_models
fi

git clone https://github.com/deepinsight/insightface.git insightface_models

# 复制模型文件
echo "复制模型文件..."
cp -r insightface_models/models/buffalo_l/det_10g.onnx models/insightface/
cp -r insightface_models/models/buffalo_l/w600k_r50.onnx models/insightface/

# 下载YOLOv8模型
echo "下载YOLOv8模型..."
if [ -d "yolov8_models" ]; then
    rm -rf yolov8_models
fi

git clone https://github.com/ultralytics/ultralytics.git yolov8_models

# 导出YOLOv8n模型
echo "导出YOLOv8n模型..."
cd yolov8_models
pip install ultralytics
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640)
"
cd ..
cp yolov8_models/yolov8n.onnx models/

# 清理
echo "清理临时文件..."
rm -rf insightface_models
rm -rf yolov8_models

echo "=== 模型下载完成 ==="
echo "模型文件位置:"
echo "  - models/insightface/det_10g.onnx"
echo "  - models/insightface/w600k_r50.onnx"
echo "  - models/yolov8n.onnx"
echo ""
echo "注意: 请确保模型文件格式正确，并与算法包兼容"