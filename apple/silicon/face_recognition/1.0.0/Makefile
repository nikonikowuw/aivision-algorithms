.PHONY: all build test validate package clean

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
        'scrfd_2.5g_bnkps',
        'AdaFace',
    ]:
        if needle not in text:
            raise SystemExit(f"algo_meta.yaml missing: {needle}")
else:
    meta = yaml.safe_load(Path("algo_meta.yaml").read_text())
    for field in ["algorithm_name", "algorithm", "version", "domain", "result_schema", "capabilities", "ai_params_schema"]:
        if field not in meta:
            raise SystemExit(f"Missing algo_meta field: {field}")
    if meta["algorithm"] != "face_recognition":
        raise SystemExit("algorithm must be face_recognition")
    props = meta["ai_params_schema"].get("properties", {})
    expected = [
        "enable_tracker",
        "person_conf_thres",
        "face_conf_thres",
        "recognition_threshold",
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
	@echo "=== Checking Models ==="
	@for model in weights/scrfd_person_2.5g.mlpackage weights/scrfd_2.5g_bnkps_shape160x160.mlpackage weights/adaface_ir101_webface4m.mlpackage; do \
		if [ ! -d "$$model" ]; then \
			echo "✗ Missing model: $$model"; \
			exit 1; \
		fi; \
	done
	@echo "=== Building Face Recognition Algorithm Package ==="
	@mkdir -p build
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build --config Release -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@if [ -f "build/src/app/nikoniko_detector.so" ]; then \
		cp build/src/app/nikoniko_detector.so ./nikoniko_detector.so; \
	fi
	@echo "✓ Output: nikoniko_detector.so"

validate:
	@echo "=== Validating Algorithm Package ==="
	@for file in algo_meta.yaml label_map.json testimage.jpg CMakeLists.txt src/app/abi_entry.mm; do \
		if [ ! -f "$$file" ]; then \
			echo "✗ Missing required file: $$file"; \
			exit 1; \
		fi; \
		echo "✓ $$file"; \
	done
	@for dir in weights/scrfd_person_2.5g.mlpackage weights/scrfd_2.5g_bnkps_shape160x160.mlpackage weights/adaface_ir101_webface4m.mlpackage; do \
		if [ ! -d "$$dir" ]; then \
			echo "✗ Missing required CoreML weights directory: $$dir"; \
			exit 1; \
		fi; \
		echo "✓ $$dir"; \
	done
	@if [ -f "nikoniko_detector.so" ]; then \
		if command -v nm >/dev/null 2>&1; then \
			if [ "$$(uname -s)" = "Linux" ]; then \
				nm_output="$$(nm -D nikoniko_detector.so 2>/dev/null || nm -g nikoniko_detector.so 2>/dev/null || true)"; \
			else \
				nm_output="$$(nm -gU nikoniko_detector.so 2>/dev/null || true)"; \
			fi; \
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
	@clang++ -std=c++17 -I../../../../engine/include tests/test_abi.cpp -ldl -o /tmp/face_recognition_test_abi
	@/tmp/face_recognition_test_abi ./nikoniko_detector.so
	@echo ""
	@echo "=== Running Algorithm Self-Test ==="
	@echo "$$TEST_SCRIPT" | python3
	@echo ""
	@echo "=== Preparing Test Image ==="
	@clang++ -std=c++17 -fobjc-arc tests/convert_image.mm -framework CoreGraphics -framework ImageIO -o /tmp/face_recognition_convert_image
	@/tmp/face_recognition_convert_image
	@rm -f /tmp/face_recognition_convert_image
	@echo ""
	@echo "=== Compiling End-to-End Inference Test ==="
	@clang++ -std=c++17 -fobjc-arc -I../../../../engine/include tests/test_e2e_infer.mm -ldl -framework Foundation -framework CoreGraphics -framework ImageIO -framework CoreText -o /tmp/face_recognition_test_e2e
	@/tmp/face_recognition_test_e2e ./nikoniko_detector.so --raw-bgr /tmp/testimage_raw.bgr
	@echo ""
	@echo "=== Testing Completed ==="

package: build validate
	@echo "=== Packaging Face Recognition Algorithm ==="
	@binary_info=$$(file -b nikoniko_detector.so 2>/dev/null || true); \
	if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! printf '%s\n' "$$binary_info" | grep -qi 'Mach-O'; then \
			echo "✗ nikoniko_detector.so is not a macOS Mach-O shared library: $$binary_info"; \
			exit 1; \
		fi; \
	else \
		if ! printf '%s\n' "$$binary_info" | grep -q 'ELF'; then \
			echo "✗ nikoniko_detector.so is not a Linux ELF shared library: $$binary_info"; \
			exit 1; \
		fi; \
	fi
	@rm -rf /tmp/face_recognition_1.0.0
	@mkdir -p /tmp/face_recognition_1.0.0/models
	@mkdir -p /tmp/face_recognition_1.0.0/weights
	@cp algo_meta.yaml label_map.json testimage.jpg nikoniko_detector.so README.md .env /tmp/face_recognition_1.0.0/
	@if [ -d "models" ]; then cp -r models/* /tmp/face_recognition_1.0.0/models/ 2>/dev/null || true; fi
	@cp -r weights/* /tmp/face_recognition_1.0.0/weights/
	@rm -f face_recognition_1.0.0.zip
	@(cd /tmp && zip -r face_recognition_1.0.0.zip face_recognition_1.0.0)
	@mv /tmp/face_recognition_1.0.0.zip ./
	@rm -rf /tmp/face_recognition_1.0.0
	@echo "✓ Output file: face_recognition_1.0.0.zip"
	@echo "=== Packaging Completed ==="

clean:
	rm -rf build nikoniko_detector.so *.zip /tmp/face_recognition_test_abi
