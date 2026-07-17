#!/usr/bin/env python3
"""Validate the exact staged Apple Silicon runtime package."""

import argparse
import json
import pathlib
import subprocess
import sys


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_nonempty(path: pathlib.Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"required nonempty file is missing: {path}")


def require_model(path: pathlib.Path) -> None:
    if not path.is_dir() or path.suffix != ".mlmodelc":
        fail(f"compiled CoreML bundle is missing: {path}")
    files = [item for item in path.rglob("*") if item.is_file() and item.stat().st_size > 0]
    if not files:
        fail(f"compiled CoreML bundle is empty: {path}")


def run(*command: str) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"command failed ({' '.join(command)}): {result.stderr.strip()}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True, type=pathlib.Path)
    parser.add_argument("--test-tool", required=True, type=pathlib.Path)
    args = parser.parse_args()
    runtime = args.runtime_dir.resolve()

    library = runtime / "tentcoo_detection.so"
    for name in ("algo_meta.yaml", "label_map.json", "README.md", "testimage.jpg"):
        require_nonempty(runtime / name)
    require_nonempty(library)
    require_nonempty(runtime / "shaders" / "image_preprocess.metallib")
    require_model(runtime / "weights" / "damoyolo_human.mlmodelc")
    require_model(runtime / "weights" / "damoyolo_cigarette.mlmodelc")
    require_nonempty(args.test_tool)

    file_output = run("file", str(library))
    if "Mach-O 64-bit" not in file_output or "arm64" not in file_output or "dynamically linked shared library" not in file_output:
        fail(f"library is not an arm64 Mach-O shared library: {file_output.strip()}")

    symbols = run("nm", "-gU", str(library))
    for symbol in ("detector_init", "detector_infer", "detector_destroy",
                   "detector_self_test", "detector_version", "detector_name",
                   "algo_free_result"):
        if symbol not in symbols:
            fail(f"required symbol is missing: {symbol}")

    prohibited = {".pt", ".onnx", ".py", ".pyc", ".mlpackage"}
    for path in runtime.rglob("*"):
        if path.is_file() and path.suffix.lower() in prohibited:
            fail(f"prohibited runtime artifact found: {path}")

    with (runtime / "label_map.json").open("r", encoding="utf-8") as handle:
        label_map = json.load(handle)
    if "14001" not in json.dumps(label_map, ensure_ascii=False):
        fail("label_map.json does not contain category_code 14001")

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for package validation") from exc
    with (runtime / "algo_meta.yaml").open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if metadata.get("algorithm") != "smoking_detection" or metadata.get("category_code") != 14001:
        fail("algo_meta.yaml identity/category fields are inconsistent")

    run(str(args.test_tool.resolve()), str(library))
    print(f"validated runtime package: {runtime}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
