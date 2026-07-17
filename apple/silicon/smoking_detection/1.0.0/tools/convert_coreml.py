#!/usr/bin/env python3
"""Convert a DAMO TorchScript export to CoreML and optionally compile it."""

from __future__ import annotations

import argparse
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

INPUT_SIZE = 640


def validate_paths(torchscript: pathlib.Path, output: pathlib.Path,
                   compiled_output: pathlib.Path | None) -> None:
    if torchscript.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("--torchscript must end in .pt or .pth")
    if output.suffix.lower() != ".mlpackage":
        raise ValueError("--output must end in .mlpackage")
    if compiled_output is not None and compiled_output.suffix.lower() != ".mlmodelc":
        raise ValueError("--compiled-output must end in .mlmodelc")
    if compiled_output is not None and output.resolve() == compiled_output.resolve():
        raise ValueError(".mlpackage and .mlmodelc outputs must be different")


def validate_coreml_spec(model: Any, input_size: int) -> None:
    spec = model.get_spec()
    inputs = list(spec.description.input)
    outputs = list(spec.description.output)
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError("CoreML DAMO model must expose exactly one input and one output")
    if inputs[0].type.WhichOneof("Type") != "imageType":
        raise RuntimeError("CoreML input must be an image feature")
    if (inputs[0].type.imageType.width != input_size or
            inputs[0].type.imageType.height != input_size):
        raise RuntimeError("CoreML image input dimensions do not match --input-size")
    if outputs[0].type.WhichOneof("Type") != "multiArrayType":
        raise RuntimeError("CoreML output must be a multi-array")
    shape = list(outputs[0].type.multiArrayType.shape)
    if len(shape) not in {2, 3} or shape[-1] != 6 or (len(shape) == 3 and shape[0] != 1):
        raise RuntimeError(
            f"CoreML output shape must be [N,6] or [1,N,6], got {shape}"
        )


def convert(torchscript: pathlib.Path, output: pathlib.Path,
            input_size: int, image_scale: float) -> None:
    if not torchscript.is_file() or torchscript.stat().st_size == 0:
        raise RuntimeError(f"TorchScript input is missing or empty: {torchscript}")
    if input_size != INPUT_SIZE:
        raise ValueError("the runtime contract requires --input-size 640")
    if image_scale <= 0.0:
        raise ValueError("--image-scale must be positive")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing CoreML package: {output}")

    try:
        import coremltools as ct  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError("conversion requires torch and coremltools") from exc

    try:
        traced = torch.jit.load(str(torchscript), map_location="cpu").eval()
    except Exception as exc:
        raise RuntimeError("failed to load the TorchScript export") from exc

    deployment_target = getattr(ct.target, "macOS13", None)
    if deployment_target is None:
        raise RuntimeError("installed coremltools does not provide the macOS13 target")
    try:
        model = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            minimum_deployment_target=deployment_target,
            inputs=[ct.ImageType(
                name="images",
                shape=(1, 3, input_size, input_size),
                color_layout=ct.colorlayout.RGB,
                scale=image_scale,
                bias=[0.0, 0.0, 0.0],
            )],
        )
        validate_coreml_spec(model, input_size)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.author = "AI Vision Inference"
        model.short_description = "DAMO-YOLO smoking detection model"
        model.version = "1.0.0"
        model.save(str(output))
    except Exception as exc:
        raise RuntimeError(
            "TorchScript-to-CoreML conversion failed; no ONNX converter fallback is supported"
        ) from exc
    if not output.is_dir() or not any(path.is_file() and path.stat().st_size > 0
                                      for path in output.rglob("*")):
        raise RuntimeError("CoreML conversion did not create a nonempty .mlpackage")


def compile_model(mlpackage: pathlib.Path, compiled_output: pathlib.Path) -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("xcrun coremlcompiler requires macOS")
    if not mlpackage.is_dir():
        raise RuntimeError(f"CoreML package is missing: {mlpackage}")
    if compiled_output.exists():
        raise RuntimeError(f"refusing to overwrite existing compiled model: {compiled_output}")
    if shutil.which("xcrun") is None:
        raise RuntimeError("xcrun is not available; install Xcode command-line tools")

    with tempfile.TemporaryDirectory(prefix="smoking-coreml-") as temporary:
        destination = pathlib.Path(temporary)
        result = subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(destination)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"xcrun coremlcompiler failed: {result.stderr.strip() or result.stdout.strip()}"
            )
        generated = destination / f"{mlpackage.stem}.mlmodelc"
        if not generated.is_dir():
            candidates = list(destination.glob("*.mlmodelc"))
            if len(candidates) != 1:
                raise RuntimeError("coremlcompiler did not produce exactly one .mlmodelc bundle")
            generated = candidates[0]
        compiled_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(generated), str(compiled_output))
    if not any(path.is_file() and path.stat().st_size > 0
               for path in compiled_output.rglob("*")):
        raise RuntimeError("compiled .mlmodelc bundle is empty")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a TorchScript DAMO export to an image-input CoreML model"
    )
    parser.add_argument("--torchscript", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--image-scale", type=float, default=1.0 / 255.0,
                        help="CoreML image-to-tensor scale; verify against ModelScope reference")
    parser.add_argument("--compiled-output", type=pathlib.Path,
                        help="Compile with xcrun to this .mlmodelc path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_paths(args.torchscript, args.output, args.compiled_output)
        convert(args.torchscript, args.output, args.input_size, args.image_scale)
        if args.compiled_output is not None:
            compile_model(args.output, args.compiled_output)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"coreml_package={args.output}")
    if args.compiled_output is not None:
        print(f"compiled_model={args.compiled_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
