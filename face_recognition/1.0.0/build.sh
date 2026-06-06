#!/bin/bash

# 人脸识别算法包构建脚本

set -e

echo "=== 构建人脸识别算法包 ==="

# 检查依赖
echo "检查依赖..."

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到cmake"
    exit 1
fi

# 检查OpenCV
if ! pkg-config --exists opencv4; then
    echo "警告: 未找到OpenCV4，尝试使用OpenCV..."
    if ! pkg-config --exists opencv; then
        echo "错误: 未找到OpenCV"
        exit 1
    fi
fi

# 检查ONNXRuntime
if [ ! -d "/usr/local/include/onnxruntime" ]; then
    echo "警告: 未找到ONNXRuntime，请确保已安装"
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build
cd build

# 配置
echo "配置CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
echo "编译..."
make -j$(nproc)

# 安装
echo "安装..."
make install

echo "=== 构建完成 ==="
echo "输出文件: build/nikoniko_detector.so"
echo ""
echo "使用方法:"
echo "1. 将模型文件放入 models/ 目录"
echo "2. 将整个目录打包为zip文件"
echo "3. 上传到平台"