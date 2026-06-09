SHELL := /bin/bash

BUILD_DIR ?= build
BUILD_TYPE ?= Release
CMAKE ?= cmake

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  JOBS ?= $(shell sysctl -n hw.ncpu)
  NM_SYMBOLS := nm -gU
else
  JOBS ?= $(shell nproc 2>/dev/null || getconf _NPROCESSORS_ONLN)
  NM_SYMBOLS := nm -D
endif

LIB_NAME := nikoniko_detector.so
BUILD_LIB := $(BUILD_DIR)/$(LIB_NAME)
TEST_BIN := /tmp/run_self_test
REQUIRED_SYMBOLS := detector_init detector_infer detector_destroy
PACKAGE_NAME := face_recognition_1.0.0

.PHONY: all configure build symbols package setup-mac-ml yolo-coreml benchmark-insightface clean distclean help

all: build

test: build
	@echo "编译自检程序..."
	/usr/bin/c++ -std=c++17 -o $(TEST_BIN) tests/test_runner.cpp -ldl
	@echo "运行自检..."
	@ALGO_PACKAGE_DIR="$(CURDIR)" $(TEST_BIN) ./$(LIB_NAME)
	@echo "✓ 自检通过"

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j$(JOBS)
	cp $(BUILD_LIB) ./$(LIB_NAME)
	@echo "输出文件: $(CURDIR)/$(LIB_NAME)"

symbols: build
	@echo "检查 ABI 符号..."
	@for symbol in $(REQUIRED_SYMBOLS); do \
		if $(NM_SYMBOLS) ./$(LIB_NAME) | grep -q "_$${symbol}\| $${symbol}"; then \
			echo "✓ $${symbol}"; \
		else \
			echo "✗ $${symbol} (必需)"; \
			exit 1; \
		fi; \
	done

package: test
	rm -rf /tmp/$(PACKAGE_NAME) /tmp/$(PACKAGE_NAME).zip
	mkdir -p /tmp/$(PACKAGE_NAME)
	cp algo_meta.yaml $(LIB_NAME) testimage.jpg label_map.json README.md /tmp/$(PACKAGE_NAME)/
	[ ! -d models ] || cp -R models /tmp/$(PACKAGE_NAME)/
	cd /tmp && zip -qr $(PACKAGE_NAME).zip $(PACKAGE_NAME)
	mv /tmp/$(PACKAGE_NAME).zip ./
	rm -rf /tmp/$(PACKAGE_NAME)
	@echo "输出文件: $(PACKAGE_NAME).zip"

setup-mac-ml:
	python3 -m venv .venv-mac-ml
	. .venv-mac-ml/bin/activate && python -m pip install --upgrade pip
	. .venv-mac-ml/bin/activate && python -m pip install 'numpy<2' coremltools onnxruntime-silicon ultralytics

yolo-coreml:
	@if [ -x .venv-mac-ml/bin/python ]; then \
		.venv-mac-ml/bin/python tools/export_yolo_coreml.py; \
	else \
		python3 tools/export_yolo_coreml.py; \
	fi

verify-insightface-coreml:
	@if [ -x .venv-mac-ml/bin/python ]; then \
		.venv-mac-ml/bin/python tools/benchmark_insightface_onnx.py; \
	else \
		python3 tools/benchmark_insightface_onnx.py; \
	fi

benchmark-insightface: verify-insightface-coreml

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean 2>/dev/null || true
	rm -f ./$(LIB_NAME)

distclean:
	rm -rf $(BUILD_DIR) ./$(LIB_NAME) ./$(PACKAGE_NAME).zip

help:
	@echo "make          编译并复制 $(LIB_NAME)"
	@echo "make symbols  编译并校验 ABI 符号"
	@echo "make package  编译、校验并打包 zip"
	@echo "make setup-mac-ml 安装 CoreML/ONNXRuntime 评估依赖"
	@echo "make yolo-coreml 导出 YOLO CoreML 模型"
	@echo "make verify-insightface-coreml 验证 InsightFace 仅使用 CoreML EP"
	@echo "make benchmark-insightface 兼容别名，等同 verify-insightface-coreml"
	@echo "make clean    清理动态库"
	@echo "make distclean 删除 build、动态库和 zip"
