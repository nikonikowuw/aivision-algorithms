.PHONY: all build validate test package clean

# ---- Python validation script (inline) ----
define VALIDATE_SCRIPT
import json
from pathlib import Path
try:
    import yaml
except ImportError:
    yaml = None

if yaml is None:
    text = Path("algo_meta.yaml").read_text()
    for needle in [
        'algorithm: "face_recognition"',
        'version: "1.0.0"',
        'result_schema: "face_identity_track"',
        'detect_person',
        'RetinaFace',
        'AdaFace',
        'nvidia_gpu',
    ]:
        if needle not in text:
            raise SystemExit(f"algo_meta.yaml missing: {needle}")
else:
    meta = yaml.safe_load(Path("algo_meta.yaml").read_text())
    for field in ["algorithm", "version", "domain", "result_schema", "capabilities", "ai_params_schema"]:
        if field not in meta:
            raise SystemExit(f"Missing algo_meta field: {field}")
    if meta["algorithm"] != "face_recognition":
        raise SystemExit("algorithm must be face_recognition")
    if "nvidia_gpu" not in meta.get("hardware", []):
        raise SystemExit("hardware must include nvidia_gpu")
    props = meta["ai_params_schema"].get("properties", {})
    expected = [
        "enable_tracker",
        "person_conf_thres",
        "face_conf_thres",
        "recognition_threshold",
        "max_batch_size",
    ]
    for key in expected:
        if key not in props:
            raise SystemExit(f"Missing parameter definition: {key}")

items = json.loads(Path("label_map.json").read_text())
codes = {item.get("category_code") for item in items}
for code in (13001, 13002, 13003):
    if code not in codes:
        raise SystemExit(f"Missing category_code: {code}")
for item in items:
    code = item.get("category_code")
    if not isinstance(code, int) or not 10000 <= code <= 99999:
        raise SystemExit(f"Invalid category_code: {code}")
print("✓ YAML/JSON content validation passed")
endef
export VALIDATE_SCRIPT

# ---- Python self-test script (inline) ----
define TEST_SCRIPT
import ctypes
import os
lib = ctypes.CDLL(os.path.abspath("./nikoniko_detector.so"))
lib.detector_self_test.restype = ctypes.c_int
ret = lib.detector_self_test()
if ret != 0:
    raise SystemExit(f"detector_self_test failed with exit code: {ret}")
print("✓ detector_self_test passed")
endef
export TEST_SCRIPT

all: build

build:
	@echo "=== Checking TensorRT Engines ==="
	@for engine in weights/yolo11n.engine weights/retinaface.engine weights/adaface_ir50.engine; do \
		if [ ! -f "$$engine" ]; then \
			echo "✗ Missing engine: $$engine"; \
			exit 1; \
		fi; \
	done
	@echo "=== Building GPU Face Recognition Algorithm Package ==="
	@mkdir -p build
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build --config Release -j$$(nproc 2>/dev/null || echo 4)
	@if [ -f "build/src/app/nikoniko_detector.so" ]; then \
		cp build/src/app/nikoniko_detector.so ./nikoniko_detector.so; \
	fi
	@echo "✓ Output: nikoniko_detector.so"

validate:
	@echo "=== Validating Algorithm Package ==="
	@for file in algo_meta.yaml label_map.json testimage.jpg CMakeLists.txt src/app/abi_entry.cpp; do \
		if [ ! -f "$$file" ]; then \
			echo "✗ Missing required file: $$file"; \
			exit 1; \
		fi; \
		echo "✓ $$file"; \
	done
	@for engine in weights/yolo11n.engine weights/retinaface.engine weights/adaface_ir50.engine; do \
		if [ ! -f "$$engine" ]; then \
			echo "✗ Missing required engine file: $$engine"; \
			exit 1; \
		fi; \
		echo "✓ $$engine"; \
	done
	@if [ -f "nikoniko_detector.so" ]; then \
		if command -v nm >/dev/null 2>&1; then \
			nm_output="$$(nm -D nikoniko_detector.so 2>/dev/null || nm -g nikoniko_detector.so 2>/dev/null || true)"; \
			for symbol in detector_init detector_infer detector_update_face_library detector_free_result detector_destroy detector_self_test detector_version detector_name; do \
				if printf '%s\n' "$$nm_output" | grep -Eq "(^|[[:space:]_])$${symbol}$$"; then \
					echo "✓ symbol: $$symbol"; \
				else \
					echo "✗ Missing symbol: $$symbol"; \
					exit 1; \
				fi; \
			done; \
		fi; \
	else \
		echo "○ nikoniko_detector.so not found, skipping symbol checks"; \
	fi
	@echo "$$VALIDATE_SCRIPT" | python3
	@echo "=== Validation Passed ==="

test: build validate
	@echo "=== Compiling ABI Test Program ==="
	@g++ -std=c++17 -I../../../../engine/include tools/test_algo.cpp -ldl -o /tmp/gpu_face_recognition_test_algo
	@/tmp/gpu_face_recognition_test_algo ./nikoniko_detector.so
	@echo ""
	@echo "=== Running Algorithm Self-Test ==="
	@echo "$$TEST_SCRIPT" | python3
	@echo ""
	@echo "=== Testing Completed ==="

package: build validate
	@echo "=== Packaging GPU Face Recognition Algorithm ==="
	@binary_info=$$(file -b nikoniko_detector.so 2>/dev/null || true); \
	if ! printf '%s\n' "$$binary_info" | grep -q 'ELF'; then \
		echo "✗ nikoniko_detector.so is not a Linux ELF shared library: $$binary_info"; \
		exit 1; \
	fi
	@rm -rf /tmp/gpu_face_recognition_1.0.0
	@mkdir -p /tmp/gpu_face_recognition_1.0.0/weights
	@mkdir -p /tmp/gpu_face_recognition_1.0.0/include
	@cp algo_meta.yaml label_map.json testimage.jpg nikoniko_detector.so README.md /tmp/gpu_face_recognition_1.0.0/
	@if [ -f ".env" ]; then cp .env /tmp/gpu_face_recognition_1.0.0/; fi
	@cp -r weights/* /tmp/gpu_face_recognition_1.0.0/weights/ 2>/dev/null || true
	@cp include/algo_api.h /tmp/gpu_face_recognition_1.0.0/include/
	@rm -f gpu_face_recognition_1.0.0.zip
	@(cd /tmp && zip -r gpu_face_recognition_1.0.0.zip gpu_face_recognition_1.0.0)
	@mv /tmp/gpu_face_recognition_1.0.0.zip ./
	@rm -rf /tmp/gpu_face_recognition_1.0.0
	@echo "✓ Output file: gpu_face_recognition_1.0.0.zip"
	@echo "=== Packaging Completed ==="

clean:
	rm -rf build nikoniko_detector.so *.zip /tmp/gpu_face_recognition_test_algo
