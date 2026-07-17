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
        'algorithm: "people_counting"',
        'version: "1.0.0"',
        'result_schema: "people_counting"',
        'detect_person',
        'track_person',
    ]:
        if needle not in text:
            raise SystemExit(f"algo_meta.yaml missing: {needle}")
else:
    meta = yaml.safe_load(Path("algo_meta.yaml").read_text())
    for field in ["algorithm", "version", "domain", "result_schema", "capabilities", "ai_params_schema"]:
        if field not in meta:
            raise SystemExit(f"Missing algo_meta field: {field}")
    if meta["algorithm"] != "people_counting":
        raise SystemExit("algorithm must be people_counting")
    props = meta["ai_params_schema"].get("properties", {})
    expected = [
        "enable_tracker",
        "conf_threshold",
        "iou_threshold",
        "track_buffer",
        "match_threshold",
    ]
    for key in expected:
        if key not in props:
            raise SystemExit(f"Missing parameter definition: {key}")

items = json.loads(Path("label_map.json").read_text())
if "10001" not in items:
    raise SystemExit("Missing category_code: 10001")
print("✓ YAML/JSON content validation passed")
endef
export VALIDATE_SCRIPT

all: build

build:
	@echo "=== Building People Counting Algorithm Package ==="
	@mkdir -p build
	@cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@cmake --build build --config Release -j$$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
	@if [ -f "build/src/app/people_counting.so" ]; then \
		cp build/src/app/people_counting.so ./people_counting.so; \
	fi
	@echo "✓ Output: people_counting.so"

validate:
	@echo "=== Validating Algorithm Package ==="
	@for file in algo_meta.yaml label_map.json testimage.jpg CMakeLists.txt src/app/algo_api.mm; do \
		if [ ! -f "$$file" ]; then \
			echo "✗ Missing required file: $$file"; \
			exit 1; \
		fi; \
		echo "✓ $$file"; \
	done
	@if [ -f "people_counting.so" ]; then \
		if command -v nm >/dev/null 2>&1; then \
			if [ "$$(uname -s)" = "Linux" ]; then \
				nm_output="$$(nm -D people_counting.so 2>/dev/null || nm -g people_counting.so 2>/dev/null || true)"; \
			else \
				nm_output="$$(nm -gU people_counting.so 2>/dev/null || true)"; \
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
		echo "○ people_counting.so not found, skipping symbol checks"; \
	fi
	@echo "$$VALIDATE_SCRIPT" | python3
	@echo "=== Validation Passed ==="

package: build validate
	@echo "=== Packaging People Counting Algorithm ==="
	@binary_info=$$(file -b people_counting.so 2>/dev/null || true); \
	if [ "$$(uname -s)" = "Darwin" ]; then \
		if ! printf '%s\n' "$$binary_info" | grep -qi 'Mach-O'; then \
			echo "✗ people_counting.so is not a macOS Mach-O shared library: $$binary_info"; \
			exit 1; \
		fi; \
	else \
		if ! printf '%s\n' "$$binary_info" | grep -q 'ELF'; then \
			echo "✗ people_counting.so is not a Linux ELF shared library: $$binary_info"; \
			exit 1; \
		fi; \
	fi
	@rm -rf /tmp/people_counting_1.0.0
	@mkdir -p /tmp/people_counting_1.0.0/weights
	@cp algo_meta.yaml label_map.json testimage.jpg people_counting.so README.md /tmp/people_counting_1.0.0/
	@if [ -d "weights" ]; then cp -r weights/* /tmp/people_counting_1.0.0/weights/ 2>/dev/null || true; fi
	@rm -f people_counting_1.0.0.zip
	@(cd /tmp && zip -r people_counting_1.0.0.zip people_counting_1.0.0)
	@mv /tmp/people_counting_1.0.0.zip ./
	@rm -rf /tmp/people_counting_1.0.0
	@echo "✓ Output file: people_counting_1.0.0.zip"
	@echo "=== Packaging Completed ==="

clean:
	rm -rf build people_counting.so *.zip
