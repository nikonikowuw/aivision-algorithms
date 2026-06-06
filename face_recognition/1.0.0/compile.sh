#!/bin/bash

# 编译脚本

set -e

echo "=== 编译人脸识别算法包 ==="

# 检查依赖
echo "检查依赖..."

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到cmake"
    exit 1
fi

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到g++"
    exit 1
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

# 复制动态库
echo "复制动态库..."
cp nikoniko_detector.so ../

# 返回原目录
cd ..

echo "=== 编译完成 ==="
echo "输出文件: nikoniko_detector.so"
echo ""
echo "现在可以运行验证脚本: ./validate.sh"