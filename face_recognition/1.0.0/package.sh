#!/bin/bash

# 算法包打包脚本

set -e

echo "=== 打包人脸识别算法包 ==="

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
    fi
done

# 检查模型目录
if [ ! -d "models" ]; then
    echo "警告: 未找到models目录"
fi

# 创建打包目录
echo "创建打包目录..."
package_name="face_recognition_1.0.0"
package_dir="/tmp/$package_name"

rm -rf "$package_dir"
mkdir -p "$package_dir"

# 复制文件
echo "复制文件..."
cp algo_meta.yaml "$package_dir/"
cp nikoniko_detector.so "$package_dir/"
cp testimage.jpg "$package_dir/"
cp label_map.json "$package_dir/"

if [ -d "models" ]; then
    cp -r models "$package_dir/"
fi

cp README.md "$package_dir/"

# 创建压缩包
echo "创建压缩包..."
cd /tmp
zip -r "$package_name.zip" "$package_name"

# 移动到当前目录
mv "$package_name.zip" /Users/niko/dev/go/aivisioninference/algorithms/face_recognition/1.0.0/

# 清理
rm -rf "$package_dir"

echo "=== 打包完成 ==="
echo "输出文件: $package_name.zip"
echo ""
echo "使用方法:"
echo "1. 上传 $package_name.zip 到平台"
echo "2. 平台会自动解压和验证"
echo "3. 验证通过后即可使用"