#!/bin/bash

# 算法包验证脚本

set -e

echo "=== 验证人脸识别算法包 ==="

# 检查必需文件
echo "检查必需文件..."

required_files=(
    "algo_meta.yaml"
    "nikoniko_detector.so"
    "testimage.jpg"
    "label_map.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "错误: 缺少以下必需文件:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo "✓ 所有必需文件都存在"

# 检查动态库符号
echo ""
echo "检查动态库符号..."

required_symbols=(
    "detector_init"
    "detector_infer"
    "detector_destroy"
)

optional_symbols=(
    "detector_version"
    "detector_name"
    "detector_self_test"
)

if command -v nm &> /dev/null; then
    for symbol in "${required_symbols[@]}"; do
        if nm -D nikoniko_detector.so | grep -q "$symbol"; then
            echo "✓ $symbol"
        else
            echo "✗ $symbol (必需)"
            exit 1
        fi
    done
    
    for symbol in "${optional_symbols[@]}"; do
        if nm -D nikoniko_detector.so | grep -q "$symbol"; then
            echo "✓ $symbol (可选)"
        else
            echo "○ $symbol (可选，未实现)"
        fi
    done
else
    echo "警告: 未找到nm命令，跳过符号检查"
fi

# 检查配置文件语法
echo ""
echo "检查配置文件语法..."

if command -v python3 &> /dev/null; then
    # 检查YAML语法
    if python3 -c "import yaml; yaml.safe_load(open('algo_meta.yaml'))" 2>/dev/null; then
        echo "✓ algo_meta.yaml 语法正确"
    else
        echo "✗ algo_meta.yaml 语法错误"
        exit 1
    fi
    
    # 检查JSON语法
    if python3 -c "import json; json.load(open('label_map.json'))" 2>/dev/null; then
        echo "✓ label_map.json 语法正确"
    else
        echo "✗ label_map.json 语法错误"
        exit 1
    fi
    
    # 检查algo_meta.yaml内容
    echo ""
    echo "检查algo_meta.yaml内容..."
    
    python3 << 'EOF'
import yaml

with open('algo_meta.yaml', 'r') as f:
    meta = yaml.safe_load(f)

required_fields = ['algorithm', 'version', 'domain', 'result_schema', 'capabilities']
for field in required_fields:
    if field not in meta:
        print(f"✗ 缺少必需字段: {field}")
        exit(1)
    else:
        print(f"✓ {field}: {meta[field]}")

# 检查capabilities
if 'image' in meta.get('capabilities', {}):
    print(f"✓ image capabilities: {meta['capabilities']['image']}")
if 'data' in meta.get('capabilities', {}):
    print(f"✓ data capabilities: {meta['capabilities']['data']}")

# 检查ai_params_schema
if 'ai_params_schema' in meta:
    schema = meta['ai_params_schema']
    if 'properties' in schema:
        print(f"✓ ai_params_schema 定义了 {len(schema['properties'])} 个参数")
    if 'required' in schema:
        print(f"✓ 必需参数: {schema['required']}")
EOF
    
    # 检查label_map.json内容
    echo ""
    echo "检查label_map.json内容..."
    
    python3 << 'EOF'
import json

with open('label_map.json', 'r') as f:
    label_map = json.load(f)

if not isinstance(label_map, list):
    print("✗ label_map.json 应该是数组")
    exit(1)

print(f"✓ label_map.json 包含 {len(label_map)} 个条目")

for i, item in enumerate(label_map):
    if 'label_id' not in item:
        print(f"✗ 条目 {i} 缺少 label_id")
        exit(1)
    if 'category_code' not in item:
        print(f"✗ 条目 {i} 缺少 category_code")
        exit(1)
    
    category_code = item['category_code']
    if not isinstance(category_code, int) or category_code < 10000 or category_code > 99999:
        print(f"✗ 条目 {i} 的 category_code 必须是5位数字")
        exit(1)
    
    print(f"✓ 条目 {i}: label_id={item['label_id']}, category_code={category_code}")
EOF
    
else
    echo "警告: 未找到python3，跳过配置文件检查"
fi

# 检查测试图片
echo ""
echo "检查测试图片..."

if [ -f "testimage.jpg" ]; then
    file_size=$(stat -f%z testimage.jpg 2>/dev/null || stat -c%s testimage.jpg 2>/dev/null)
    if [ "$file_size" -gt 0 ]; then
        echo "✓ testimage.jpg 存在且大小为 $file_size 字节"
    else
        echo "✗ testimage.jpg 大小为0"
        exit 1
    fi
else
    echo "✗ testimage.jpg 不存在"
    exit 1
fi

echo ""
echo "=== 验证通过 ==="
echo ""
echo "算法包符合规范，可以上传到平台。"