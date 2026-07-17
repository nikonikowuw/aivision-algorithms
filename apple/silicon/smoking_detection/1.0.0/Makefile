BUILD_DIR := build
RUNTIME_DIR := $(BUILD_DIR)/runtime
PACKAGE_NAME := smoking_detection_1.0.0

.PHONY: all build test validate package clean

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --config Release -j$$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

validate: build
	python3 tools/validate_package.py --runtime-dir $(RUNTIME_DIR) --test-tool $(BUILD_DIR)/bin/smoking_test_tool

package: validate
	rm -rf $(BUILD_DIR)/package
	mkdir -p $(BUILD_DIR)/package/$(PACKAGE_NAME)
	cp -R $(RUNTIME_DIR)/. $(BUILD_DIR)/package/$(PACKAGE_NAME)/
	cd $(BUILD_DIR)/package && zip -r ../$(PACKAGE_NAME).zip $(PACKAGE_NAME)

clean:
	rm -rf $(BUILD_DIR)
