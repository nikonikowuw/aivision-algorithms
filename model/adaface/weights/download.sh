#!/bin/bash

# 下载链接
# WebFace12M https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing
# WebFace4M https://drive.google.com/file/d/18jQkqB0avFqWa0Pas52g54xNshUOQJpQ/view?usp=sharing
set -euo pipefail

# 用法: ./download.sh [n|s|m] (默认: n)
MODEL="${1:-n}"

# 只允许 n, s, m
if [[ ! "$MODEL" =~ ^[nsm]$ ]]; then
    echo "❌ 错误: 只支持 n, s, m"
    echo "用法: $0 [n|s|m]"
    exit 1
fi

FILE_NAME="yolo11${MODEL}.pt"
URL="https://drive.google.com/file/d/1dswnavflETcnAuplZj1IOKKP0eM8ITgT/view?usp=sharing"
# 是否跳过证书验证
SKIP_CERT="${INSECURE:-false}"

echo "📥 准备下载 ${FILE_NAME} ..."

download_success=false

if command -v curl >/dev/null 2>&1; then
    echo "⚡️ 使用 curl 下载..."
    CURL_OPTS=("-L" "-o" "${FILE_NAME}")
    if [ "$SKIP_CERT" = "true" ]; then
        CURL_OPTS+=("-k")
    fi
    # 捕获 curl 的退出码而不直接退出
    set +e
    curl "${CURL_OPTS[@]}" "$URL"
    CURL_EXIT=$?
    set -e
    if [ $CURL_EXIT -eq 0 ]; then
        download_success=true
    elif [ $CURL_EXIT -eq 60 ]; then
        echo "❌ SSL 证书验证失败 (curl exit code 60)。"
        echo "💡 提示：如果您信任源，可以通过设置环境变量 INSECURE=true 跳过证书校验，例如："
        echo "   INSECURE=true $0 $MODEL"
        exit 60
    else
        echo "❌ curl 下载失败，退出码: $CURL_EXIT"
    fi
elif command -v wget >/dev/null 2>&1; then
    echo "⚡️ 使用 wget 下载..."
    WGET_OPTS=("-O" "${FILE_NAME}")
    if [ "$SKIP_CERT" = "true" ]; then
        WGET_OPTS+=("--no-check-certificate")
    fi
    set +e
    wget "${WGET_OPTS[@]}" "$URL"
    WGET_EXIT=$?
    set -e
    if [ $WGET_EXIT -eq 0 ]; then
        download_success=true
    elif [ $WGET_EXIT -eq 5 ]; then
        echo "❌ SSL 证书验证失败 (wget exit code 5)。"
        echo "💡 提示：如果您信任源，可以通过设置环境变量 INSECURE=true 跳过证书校验，例如："
        echo "   INSECURE=true $0 $MODEL"
        exit 5
    else
        echo "❌ wget 下载失败，退出码: $WGET_EXIT"
    fi
else
    echo "❌ 错误: 系统中未找到 curl 或 wget，请先安装其中之一。"
    exit 1
fi

if [ "$download_success" = "true" ]; then
    echo "✅ 完成! 文件已保存为 ${FILE_NAME}"
else
    echo "❌ 下载失败！"
    exit 1
fi

