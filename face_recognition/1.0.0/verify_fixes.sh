#!/bin/bash

# 验证修复脚本

set -e

echo "=== 验证算法包修复 ==="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查函数
check_pass() {
    echo -e "${GREEN}✅ $1${NC}"
}

check_fail() {
    echo -e "${RED}❌ $1${NC}"
    FAILED=1
}

check_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

FAILED=0

# 1. 检查必需文件
echo "1. 检查必需文件"
for file in algo_meta.yaml label_map.json testimage.jpg src/face_recognition.cpp include/face_recognition.h; do
    if [ -f "$file" ]; then
        check_pass "$file 存在"
    else
        check_fail "$file 不存在"
    fi
done
echo ""

# 2. 检查 ABI 接口导出
echo "2. 检查 ABI 接口导出"

# 项目当前接口
for symbol in detector_init detector_infer detector_free_result detector_destroy; do
    if grep -q "^.*$symbol.*(" src/face_recognition.cpp; then
        check_pass "$symbol 已定义"
    else
        check_fail "$symbol 未定义"
    fi
done

# 可选符号
for symbol in detector_version detector_name detector_self_test; do
    if grep -q "^.*$symbol.*(" src/face_recognition.cpp; then
        check_pass "$symbol (可选) 已定义"
    else
        check_warn "$symbol (可选) 未定义"
    fi
done

# PRD 接口（向后兼容）
for symbol in create_detector detector_infer_v2 detector_free_result_v2 destroy_detector; do
    if grep -q "^.*$symbol.*(" src/face_recognition.cpp; then
        check_pass "$symbol (PRD接口) 已定义"
    else
        check_warn "$symbol (PRD接口) 未定义"
    fi
done
echo ""

# 3. 检查 extern "C" 块
echo "3. 检查 extern \"C\" 块"
if grep -q 'extern "C" {' src/face_recognition.cpp; then
    check_pass "extern \"C\" 块存在"
    
    # 检查所有 ABI 函数是否在 extern "C" 块中
    EXTERN_START=$(grep -n 'extern "C" {' src/face_recognition.cpp | head -1 | cut -d: -f1)
    EXTERN_END=$(grep -n '} // extern "C"' src/face_recognition.cpp | head -1 | cut -d: -f1)
    
    if [ -n "$EXTERN_START" ] && [ -n "$EXTERN_END" ]; then
        check_pass "extern \"C\" 块范围: 行 $EXTERN_START - $EXTERN_END"
    else
        check_fail "无法确定 extern \"C\" 块范围"
    fi
else
    check_fail "extern \"C\" 块不存在"
fi
echo ""

# 4. 检查 dma_fd 修复
echo "4. 检查 dma_fd 内存访问修复"
if grep -q "memcpy.*dma_fd" src/face_recognition.cpp; then
    check_fail "仍然存在 memcpy(dma_fd) 错误"
else
    check_pass "已移除 memcpy(dma_fd) 错误"
fi

if grep -q "mmap.*dma_fd" src/face_recognition.cpp; then
    check_pass "使用 mmap 访问 DMA 缓冲区"
else
    check_warn "未找到 mmap 访问"
fi
echo ""

# 5. 检查参数验证
echo "5. 检查参数验证"
if grep -q "input->width == 0 || input->height == 0" src/face_recognition.cpp; then
    check_pass "存在尺寸验证"
else
    check_warn "缺少尺寸验证"
fi

if grep -q "input->width > 8192" src/face_recognition.cpp; then
    check_pass "存在尺寸上限验证"
else
    check_warn "缺少尺寸上限验证"
fi
echo ""

# 6. 检查可配置阈值
echo "6. 检查可配置阈值"
if grep -q "recognition_threshold" src/face_recognition.cpp; then
    check_pass "recognition_threshold 已实现"
else
    check_fail "recognition_threshold 未实现"
fi

if grep -q "recognition_threshold" include/face_recognition.h; then
    check_pass "recognition_threshold 已添加到配置结构"
else
    check_fail "recognition_threshold 未添加到配置结构"
fi

if grep -q "recognition_threshold" algo_meta.yaml; then
    check_pass "recognition_threshold 已添加到 algo_meta.yaml"
else
    check_fail "recognition_threshold 未添加到 algo_meta.yaml"
fi
echo ""

# 7. 检查错误码
echo "7. 检查错误码"
if grep -q "enum class AlgoError" src/face_recognition.cpp; then
    check_pass "错误码枚举已定义"
else
    check_warn "缺少错误码枚举"
fi
echo ""

# 8. 检查日志
echo "8. 检查日志"
if grep -q "ALGO_LOG_ERROR" src/face_recognition.cpp; then
    check_pass "存在错误日志"
else
    check_warn "缺少错误日志"
fi

if grep -q "ALGO_LOG_INFO" src/face_recognition.cpp; then
    check_pass "存在信息日志"
else
    check_warn "缺少信息日志"
fi
echo ""

# 总结
echo "=== 验证完成 ==="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}所有检查通过！${NC}"
    exit 0
else
    echo -e "${RED}存在失败的检查项，请修复后重试${NC}"
    exit 1
fi
