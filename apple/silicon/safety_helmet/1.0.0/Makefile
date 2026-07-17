.PHONY: all build validate clean package

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
        'algorithm: "safety_helmet"',
        'version: "1.0.0"',
        'result_schema: "object_detection"',
        'detect_safety_helmet',
    ]:
        if needle not in text:
            raise SystemExit(f"algo_meta.yaml missing: {needle}")
else:
    meta = yaml.safe_load(Path("algo_meta.yaml").read_text())
    for field in ["algorithm", "version", "domain", "result_schema", "capabilities", "ai_params_schema"]:
        if field not in meta:
            raise SystemExit(f"Missing algo_meta field: {field}")
    if meta["algorithm"] != "safety_helmet":
        raise SystemExit("algorithm must be safety_helmet")
    props = meta["ai_params_schema"].get("properties", {})
    expected = [
        "conf_threshold",
        "iou_threshold",
    ]
    for key in expected:
        if key not in props:
            raise SystemExit(f"Missing parameter definition: {key}")

items = json.loads(Path("label_map.json").read_text())
if "10001" not in items:
    raise SystemExit("Missing category_code: 10001")
if "10002" not in items:
    raise SystemExit("Missing category_code: 10002")
print("✓ YAML/JSON content validation passed")
endef
export VALIDATE_SCRIPT

all: build

build:
	@echo "=== Building Safety Helmet Algorithm Package ==="
	@mkdir -p build
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build --config Release -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@if [ -f "build/src/app/tentcoo_detection.so" ]; then \
		cp build/src/app/tentcoo_detection.so ./tentcoo_detection.so; \
	fi
	@echo "✓ Output: tentcoo_detection.so"

validate:
	@echo "=== Validating Algorithm Package ==="
	@for file in algo_meta.yaml label_map.json testimage.jpg CMakeLists.txt src/app/algo_api.cpp README.md; do \
		if [ ! -f "$$file" ]; then \
			echo "✗ Missing required file: $$file"; \
			exit 1; \
		fi; \
		echo "✓ $$file"; \
	done
	@if [ -f "tentcoo_detection.so" ]; then \
		if command -v nm >/dev/null 2>&1; then \
			if [ "$$(uname -s)" = "Linux" ]; then \
				nm_output="$$(nm -D tentcoo_detection.so 2>/dev/null || nm -g tentcoo_detection.so 2>/dev/null || true)"; \
			else \
				nm_output="$$(nm -gU tentcoo_detection.so 2>/dev/null || true)"; \
			fi; \
			for symbol in detector_init detector_infer detector_destroy detector_self_test detector_version detector_name; do \
				if printf '%s\n' "$$nm_output" | grep -Eq "(^|[[:space:]_])$${symbol}$$"; then \
					echo "✓ symbol: $$symbol"; \
				else \
					echo "✗ Missing symbol: $$symbol"; \
					exit 1; \
				fi; \
			done; \
		fi; \
	else \
		echo "○ tentcoo_detection.so not found, skipping symbol checks"; \
	fi
	@echo "$$VALIDATE_SCRIPT" | python3
	@echo "=== Validation Passed ==="

package: build validate
	@echo "=== Packaging Safety Helmet Algorithm ==="
	@binary_info=$$(file -b tentcoo_detection.so 2>/dev/null || true); \
	if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! printf '%s\n' "$$binary_info" | grep -qi 'Mach-O'; then \
			echo "✗ tentcoo_detection.so is not a macOS Mach-O shared library: $$binary_info"; \
			exit 1; \
		fi; \
	else \
		if ! printf '%s\n' "$$binary_info" | grep -q 'ELF'; then \
			echo "✗ tentcoo_detection.so is not a Linux ELF shared library: $$binary_info"; \
			exit 1; \
		fi; \
	fi
	@rm -rf /tmp/safety_helmet_1.0.0
	@mkdir -p /tmp/safety_helmet_1.0.0/weights
	@cp algo_meta.yaml label_map.json testimage.jpg tentcoo_detection.so README.md /tmp/safety_helmet_1.0.0/
	@if [ -d "weights" ]; then cp -r weights/* /tmp/safety_helmet_1.0.0/weights/ 2>/dev/null || true; fi
	@rm -f safety_helmet_1.0.0.zip
	@(cd /tmp && zip -r safety_helmet_1.0.0.zip safety_helmet_1.0.0)
	@mv /tmp/safety_helmet_1.0.0.zip ./
	@rm -rf /tmp/safety_helmet_1.0.0
	@echo "✓ Output file: safety_helmet_1.0.0.zip"
	@echo "=== Packaging Completed ==="

clean:
	rm -rf build tentcoo_detection.so *.zip
