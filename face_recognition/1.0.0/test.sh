#!/bin/bash

# 测试脚本

set -e

echo "=== 测试人脸识别算法包 ==="

# 检查必需文件
echo "检查必需文件..."

required_files=(
    "algo_meta.yaml"
    "nikoniko_detector.so"
    "testimage.jpg"
    "label_map.json"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "错误: 缺少必需文件 $file"
        exit 1
    else
        echo "✓ $file"
    fi
done

# 检查模型文件
echo ""
echo "检查模型文件..."

if [ -d "models" ]; then
    if [ -f "models/insightface/det_10g.onnx" ]; then
        echo "✓ models/insightface/det_10g.onnx"
    else
        echo "✗ models/insightface/det_10g.onnx"
    fi
    
    if [ -f "models/insightface/w600k_r50.onnx" ]; then
        echo "✓ models/insightface/w600k_r50.onnx"
    else
        echo "✗ models/insightface/w600k_r50.onnx"
    fi
    
    if [ -f "models/yolov8n.onnx" ]; then
        echo "✓ models/yolov8n.onnx"
    else
        echo "✗ models/yolov8n.onnx"
    fi
else
    echo "警告: 未找到models目录"
fi

# 检查动态库
echo ""
echo "检查动态库..."

if [ -f "nikoniko_detector.so" ]; then
    echo "✓ nikoniko_detector.so"
    
    # 检查符号
    if command -v nm &> /dev/null; then
        echo "检查符号..."
        if nm -D nikoniko_detector.so | grep -q "detector_init"; then
            echo "✓ detector_init"
        else
            echo "✗ detector_init"
        fi
        
        if nm -D nikoniko_detector.so | grep -q "detector_infer"; then
            echo "✓ detector_infer"
        else
            echo "✗ detector_infer"
        fi
        
        if nm -D nikoniko_detector.so | grep -q "detector_destroy"; then
            echo "✓ detector_destroy"
        else
            echo "✗ detector_destroy"
        fi
    fi
else
    echo "✗ nikoniko_detector.so"
fi

# 检查配置文件
echo ""
echo "检查配置文件..."

if [ -f "algo_meta.yaml" ]; then
    echo "✓ algo_meta.yaml"
    
    # 检查YAML语法
    if command -v python3 &> /dev/null; then
        if python3 -c "import yaml; yaml.safe_load(open('algo_meta.yaml'))" 2>/dev/null; then
            echo "✓ YAML语法正确"
        else
            echo "✗ YAML语法错误"
        fi
    fi
fi

if [ -f "label_map.json" ]; then
    echo "✓ label_map.json"
    
    # 检查JSON语法
    if command -v python3 &> /dev/null; then
        if python3 -c "import json; json.load(open('label_map.json'))" 2>/dev/null; then
            echo "✓ JSON语法正确"
        else
            echo "✗ JSON语法错误"
        fi
    fi
fi

# 检查测试图片
echo ""
echo "检查测试图片..."

if [ -f "testimage.jpg" ]; then
    echo "✓ testimage.jpg"
    
    # 检查文件大小
    file_size=$(stat -f%z testimage.jpg 2>/dev/null || stat -c%s testimage.jpg 2>/dev/null)
    echo "  文件大小: $file_size 字节"
else
    echo "✗ testimage.jpg"
fi

echo ""
echo "=== 测试完成 ==="
echo ""
echo "如果所有检查都通过，算法包可以上传到平台。"
echo "如果有任何检查失败，请修复后重新测试。"