.PHONY: all build test validate clean package package-platform-check
.NOTPARALLEL: package

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

define PACKAGE_SCRIPT
import platform
import shutil
import subprocess
from pathlib import Path

root = Path.cwd().resolve()
binary = root / "tentcoo_detection.so"
weight = root / "weights" / "damoyolo_safety_helmet.onnx"
if platform.system() != "Darwin" or platform.machine() != "arm64":
    raise SystemExit("package is supported only on Darwin arm64")
if not weight.is_file() or weight.stat().st_size == 0:
    raise SystemExit("weights/damoyolo_safety_helmet.onnx is missing or empty")
file_info = subprocess.check_output(["file", "-b", str(binary)], text=True).strip()
if "Mach-O" not in file_info or "arm64" not in file_info:
    raise SystemExit(f"tentcoo_detection.so must be an arm64 Mach-O binary: {file_info}")

packaged_binaries = set()
pending = [binary]
while pending:
    current = pending.pop()
    if current in packaged_binaries:
        continue
    packaged_binaries.add(current)
    output = subprocess.check_output(["otool", "-L", str(current)], text=True)
    install_id_output = subprocess.check_output(
        ["otool", "-D", str(current)], text=True).splitlines()[1:]
    install_ids = {line.strip() for line in install_id_output}
    for line in output.splitlines()[1:]:
        dependency = line.strip().split(" (compatibility", 1)[0]
        if dependency in install_ids:
            continue
        if dependency.startswith("/System/Library/") or dependency.startswith("/usr/lib/"):
            continue
        if not dependency.startswith("@loader_path/"):
            raise SystemExit(f"non-package dependency in {current.name}: {dependency}")
        candidate = (current.parent / dependency[len("@loader_path/"):]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            raise SystemExit(f"@loader_path dependency escapes package root: {dependency}")
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise SystemExit(f"packaged dependency is missing or empty: {candidate}")
        pending.append(candidate)

staging = Path("/tmp/safety_helmet_1.0.0")
shutil.rmtree(staging, ignore_errors=True)
(staging / "weights").mkdir(parents=True)
for name in ["algo_meta.yaml", "label_map.json", "testimage.jpg",
             "tentcoo_detection.so", "README.md"]:
    shutil.copy2(root / name, staging / name)
shutil.copy2(weight, staging / "weights" / weight.name)
for dependency in packaged_binaries - {binary}:
    relative = dependency.relative_to(root)
    destination = staging / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dependency, destination)
archive = root / "safety_helmet_1.0.0.zip"
archive.unlink(missing_ok=True)
shutil.make_archive(str(archive.with_suffix("")), "zip", staging.parent, staging.name)
shutil.rmtree(staging)
print(f"Output file: {archive.name}")
endef
export PACKAGE_SCRIPT

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

test: build
	@echo "=== Running Self-Test ==="
	@if [ ! -f "tentcoo_detection.so" ]; then \
		echo "✗ tentcoo_detection.so not found. Run 'make build' first."; \
		exit 1; \
	fi
	@ctest --test-dir build --output-on-failure
	@if [ -f "build/test_algo" ]; then \
		build/test_algo; \
	else \
		echo "✗ test_algo not found in build/"; \
		exit 1; \
	fi
	@echo "=== Self-Test Passed ==="

validate:
	@echo "=== Validating Algorithm Package ==="
	@for file in algo_meta.yaml label_map.json testimage.jpg CMakeLists.txt \
		src/app/algo_api.cpp src/app/algo_api.mm \
		src/common/input_buffer_validator.h src/common/input_buffer_validator.cpp README.md; do \
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

package-platform-check:
	@rm -f safety_helmet_1.0.0.zip
	@if [ "$$(uname -s)" != "Darwin" ] || [ "$$(uname -m)" != "arm64" ]; then \
		echo "✗ Packaging is supported only on Darwin arm64."; \
		exit 1; \
	fi

package: package-platform-check test validate
	@echo "=== Packaging Safety Helmet Algorithm ==="
	@echo "$$PACKAGE_SCRIPT" | python3
	@echo "=== Packaging Completed ==="

clean:
	rm -rf build tentcoo_detection.so *.zip
