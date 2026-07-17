#!/usr/bin/env python3
"""Export a pinned ModelScope DAMO-YOLO model to ONNX and TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
from typing import Any

MODEL_IDS = {
    "human": "iic/cv_tinynas_human-detection_damoyolo",
    "cigarette": "iic/cv_tinynas_object-detection_damoyolo_cigarette",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
INPUT_SIZE = 640


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pin(revision: str, weight_file: str, expected_sha256: str) -> None:
    if not revision.strip() or revision.lower() in {"master", "main", "latest"}:
        raise ValueError("--revision must be an explicit immutable tag or commit")
    if not weight_file.strip() or pathlib.PurePosixPath(weight_file).is_absolute():
        raise ValueError("--weight-file must be a nonempty path relative to the model directory")
    if ".." in pathlib.PurePosixPath(weight_file).parts:
        raise ValueError("--weight-file must not escape the downloaded model directory")
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")


def validate_output_paths(onnx_output: pathlib.Path,
                          torchscript_output: pathlib.Path) -> None:
    if onnx_output.suffix.lower() != ".onnx":
        raise ValueError("--onnx-output must end in .onnx")
    if torchscript_output.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("--torchscript-output must end in .pt or .pth")
    if onnx_output.resolve() == torchscript_output.resolve():
        raise ValueError("ONNX and TorchScript outputs must be different files")


def model_directory(model: Any) -> pathlib.Path:
    value = getattr(model, "model_dir", None)
    if not value:
        raise RuntimeError("ModelScope model does not expose model_dir; cannot verify the pinned weight")
    path = pathlib.Path(value).resolve()
    if not path.is_dir():
        raise RuntimeError(f"ModelScope model_dir is not a directory: {path}")
    return path


def export_model(model_type: str, revision: str, weight_file: str,
                 expected_sha256: str, onnx_output: pathlib.Path,
                 torchscript_output: pathlib.Path, opset: int) -> None:
    validate_pin(revision, weight_file, expected_sha256)
    validate_output_paths(onnx_output, torchscript_output)
    if opset < 11 or opset > 18:
        raise ValueError("--opset must be in [11, 18]")

    try:
        import onnx  # type: ignore
        import torch  # type: ignore
        from modelscope.models import Model  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "export requires torch, onnx, and modelscope in the conversion environment"
        ) from exc

    model_id = MODEL_IDS[model_type]
    model = Model.from_pretrained(model_id, revision=revision)
    model.eval()
    if not hasattr(model, "onnx_export"):
        raise RuntimeError("the downloaded DAMO model does not expose onnx_export")
    model.onnx_export = True

    root = model_directory(model)
    weight_path = (root / weight_file).resolve()
    try:
        weight_path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("resolved weight path escapes the downloaded model directory") from exc
    if not weight_path.is_file() or weight_path.stat().st_size == 0:
        raise RuntimeError(f"pinned model weight is missing or empty: {weight_path}")
    actual_sha256 = sha256_file(weight_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"weight SHA256 mismatch for {weight_path}: expected {expected_sha256.lower()}, "
            f"got {actual_sha256.lower()}"
        )

    onnx_output.parent.mkdir(parents=True, exist_ok=True)
    torchscript_output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)

    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, sample, strict=False, check_trace=False)
            traced = torch.jit.freeze(traced.eval())
            traced.save(str(torchscript_output))
    except Exception as exc:
        raise RuntimeError(
            "TorchScript export failed; this model graph cannot enter the supported "
            "CoreML path without an explicit export wrapper"
        ) from exc
    if not torchscript_output.is_file() or torchscript_output.stat().st_size == 0:
        raise RuntimeError("TorchScript export did not create a nonempty file")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                sample,
                str(onnx_output),
                export_params=True,
                do_constant_folding=True,
                opset_version=opset,
                input_names=["images"],
                output_names=["outputs"],
                dynamic_axes=None,
            )
        checked = onnx.load(str(onnx_output))
        onnx.checker.check_model(checked, full_check=True)
    except Exception as exc:
        raise RuntimeError("fixed-shape ONNX export or onnx.checker validation failed") from exc
    if not onnx_output.is_file() or onnx_output.stat().st_size == 0:
        raise RuntimeError("ONNX export did not create a nonempty file")

    print(f"model_id={model_id}")
    print(f"revision={revision}")
    print(f"weight={weight_file}")
    print(f"sha256={actual_sha256}")
    print(f"onnx={onnx_output}")
    print(f"torchscript={torchscript_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a pinned DAMO-YOLO model to fixed-shape ONNX and TorchScript"
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--revision", required=True,
                        help="Immutable ModelScope revision tag or commit")
    parser.add_argument("--weight-file", required=True,
                        help="Weight path relative to model.model_dir")
    parser.add_argument("--sha256", required=True,
                        help="Expected SHA256 of --weight-file")
    parser.add_argument("--onnx-output", required=True, type=pathlib.Path)
    parser.add_argument("--torchscript-output", required=True, type=pathlib.Path)
    parser.add_argument("--opset", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        export_model(args.model, args.revision, args.weight_file, args.sha256,
                     args.onnx_output, args.torchscript_output, args.opset)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
