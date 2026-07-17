#!/usr/bin/env python3
"""Fail-closed PyTorch, ONNX Runtime, and CoreML parity verification."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import platform
import re
import sys
from dataclasses import dataclass
from typing import Any, Sequence

INPUT_SIZE = 640
MODEL_IDS = {
    "human": "iic/cv_tinynas_human-detection_damoyolo",
    "cigarette": "iic/cv_tinynas_object-detection_damoyolo_cigarette",
}
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


@dataclass(frozen=True)
class Match:
    reference_index: int
    candidate_index: int
    iou: float
    score_difference: float


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pin(revision: str, weight_file: str, expected_sha256: str) -> None:
    if not revision.strip() or revision.lower() in {"main", "master", "latest"}:
        raise ValueError("--revision must be an explicit immutable tag or commit")
    relative = pathlib.PurePosixPath(weight_file)
    if not weight_file.strip() or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("--weight-file must stay inside the downloaded model directory")
    if not SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")


def validate_thresholds(iou_threshold: float, score_threshold: float,
                        min_match_ratio: float, max_count_delta: int) -> None:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in [0, 1]")
    if score_threshold < 0.0:
        raise ValueError("--score-threshold must be nonnegative")
    if not 0.0 <= min_match_ratio <= 1.0:
        raise ValueError("--min-match-ratio must be in [0, 1]")
    if max_count_delta < 0:
        raise ValueError("--max-count-delta must be nonnegative")


def decode_damoyolo(output: Any, confidence_threshold: float) -> list[Detection]:
    import numpy as np

    array = np.asarray(output)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"DAMO batch dimension must be 1, got {array.shape}")
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"DAMO output shape must be [N,6] or [1,N,6], got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("DAMO output contains NaN or Infinity")
    if confidence_threshold < 0.0 or confidence_threshold > 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")

    detections: list[Detection] = []
    for row_index, row in enumerate(array):
        x1, y1, x2, y2, score, class_value = map(float, row)
        rounded_class = round(class_value)
        if abs(class_value - rounded_class) > 1.0e-4 or rounded_class < 0:
            raise ValueError(f"row {row_index} has invalid class_id {class_value}")
        if score < 0.0 or score > 1.0:
            raise ValueError(f"row {row_index} has score outside [0,1]: {score}")
        if x2 < x1 or y2 < y1:
            raise ValueError(f"row {row_index} has inverted box coordinates")
        if score >= confidence_threshold and x2 > x1 and y2 > y1:
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                score=score, class_id=int(rounded_class),
            ))
    return sorted(detections, key=lambda detection: detection.score, reverse=True)


def compute_iou(left: Detection, right: Detection) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def match_detections(reference: Sequence[Detection],
                     candidate: Sequence[Detection],
                     iou_threshold: float) -> list[Match]:
    used: set[int] = set()
    matches: list[Match] = []
    for reference_index, expected in enumerate(reference):
        best_index = -1
        best_iou = iou_threshold
        for candidate_index, actual in enumerate(candidate):
            if candidate_index in used or actual.class_id != expected.class_id:
                continue
            iou = compute_iou(expected, actual)
            if iou >= best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index >= 0:
            used.add(best_index)
            matches.append(Match(
                reference_index=reference_index,
                candidate_index=best_index,
                iou=best_iou,
                score_difference=abs(expected.score - candidate[best_index].score),
            ))
    return matches


def assert_parity(reference_name: str, reference: Sequence[Detection],
                  candidate_name: str, candidate: Sequence[Detection],
                  iou_threshold: float, score_threshold: float,
                  min_match_ratio: float, max_count_delta: int) -> list[Match]:
    if abs(len(reference) - len(candidate)) > max_count_delta:
        raise RuntimeError(
            f"{reference_name}/{candidate_name} detection count delta is "
            f"{abs(len(reference) - len(candidate))}, limit is {max_count_delta}"
        )
    matches = match_detections(reference, candidate, iou_threshold)
    denominator = max(len(reference), len(candidate), 1)
    ratio = len(matches) / denominator
    if ratio < min_match_ratio:
        raise RuntimeError(
            f"{reference_name}/{candidate_name} match ratio {ratio:.4f} is below "
            f"{min_match_ratio:.4f}"
        )
    excessive = [match for match in matches
                 if match.score_difference > score_threshold]
    if excessive:
        worst = max(excessive, key=lambda match: match.score_difference)
        raise RuntimeError(
            f"{reference_name}/{candidate_name} score difference "
            f"{worst.score_difference:.6f} exceeds {score_threshold:.6f}"
        )
    return matches


def load_preprocessed_image(path: pathlib.Path, input_size: int) -> tuple[Any, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"real parity image is missing or empty: {path}")
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("parity image loading requires Pillow and numpy") from exc

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise RuntimeError("parity image dimensions are invalid")
    scale = min(input_size / width, input_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((resized_width, resized_height), resampling)
    canvas = Image.new("RGB", (input_size, input_size), (114, 114, 114))
    canvas.paste(resized, ((input_size - resized_width) // 2,
                           (input_size - resized_height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = array.transpose(2, 0, 1)[None, ...]
    return canvas, tensor


def extract_numpy_output(output: Any) -> Any:
    if isinstance(output, dict):
        if len(output) != 1:
            raise RuntimeError(f"expected one model output, got keys {list(output)}")
        output = next(iter(output.values()))
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise RuntimeError(f"expected one model output, got {len(output)}")
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    return output


def verify_weight(model: Any, weight_file: str, expected_sha256: str) -> None:
    model_dir_value = getattr(model, "model_dir", None)
    if not model_dir_value:
        raise RuntimeError("ModelScope model does not expose model_dir")
    root = pathlib.Path(model_dir_value).resolve()
    weight = (root / weight_file).resolve()
    try:
        weight.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("resolved weight path escapes model_dir") from exc
    if not weight.is_file() or weight.stat().st_size == 0:
        raise RuntimeError(f"pinned weight is missing or empty: {weight}")
    actual = sha256_file(weight)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"weight SHA256 mismatch: expected {expected_sha256.lower()}, got {actual.lower()}"
        )


def run_pytorch(model_id: str, revision: str, weight_file: str,
                expected_sha256: str, tensor: Any,
                confidence_threshold: float) -> list[Detection]:
    try:
        import torch
        from modelscope.models import Model
    except ImportError as exc:
        raise RuntimeError("PyTorch parity requires torch and modelscope") from exc
    model = Model.from_pretrained(model_id, revision=revision)
    verify_weight(model, weight_file, expected_sha256)
    model.eval()
    if not hasattr(model, "onnx_export"):
        raise RuntimeError("DAMO model does not expose onnx_export")
    model.onnx_export = True
    with torch.no_grad():
        output = model(torch.from_numpy(tensor))
    return decode_damoyolo(extract_numpy_output(output), confidence_threshold)


def run_onnx(path: pathlib.Path, tensor: Any,
             confidence_threshold: float) -> list[Detection]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"ONNX model is missing or empty: {path}")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX parity requires onnx and onnxruntime") from exc
    checked = onnx.load(str(path))
    onnx.checker.check_model(checked, full_check=True)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise RuntimeError("ONNX DAMO model must expose exactly one input and one output")
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    return decode_damoyolo(output, confidence_threshold)


def run_coreml(path: pathlib.Path, image: Any,
               confidence_threshold: float) -> list[Detection]:
    if platform.system() != "Darwin":
        raise RuntimeError("CoreML parity must run on macOS")
    if not path.exists():
        raise RuntimeError(f"CoreML model is missing: {path}")
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError("CoreML parity requires coremltools") from exc
    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
    spec = model.get_spec()
    if len(spec.description.input) != 1 or len(spec.description.output) != 1:
        raise RuntimeError("CoreML DAMO model must expose exactly one input and one output")
    input_feature = spec.description.input[0]
    output_feature = spec.description.output[0]
    if input_feature.type.WhichOneof("Type") != "imageType":
        raise RuntimeError("CoreML parity model input is not an image")
    if output_feature.type.WhichOneof("Type") != "multiArrayType":
        raise RuntimeError("CoreML parity model output is not a multi-array")
    prediction = model.predict({input_feature.name: image})
    if output_feature.name not in prediction or len(prediction) != 1:
        raise RuntimeError("CoreML prediction did not return the declared sole output")
    return decode_damoyolo(prediction[output_feature.name], confidence_threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Require and compare PyTorch, ONNX Runtime, and CoreML DAMO results"
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weight-file", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--onnx", required=True, type=pathlib.Path)
    parser.add_argument("--coreml", required=True, type=pathlib.Path)
    parser.add_argument("--image", required=True, type=pathlib.Path)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--iou-threshold", type=float, default=0.98)
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--min-match-ratio", type=float, default=1.0)
    parser.add_argument("--max-count-delta", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_pin(args.revision, args.weight_file, args.sha256)
        validate_thresholds(args.iou_threshold, args.score_threshold,
                            args.min_match_ratio, args.max_count_delta)
        if args.input_size != INPUT_SIZE:
            raise ValueError("the runtime contract requires --input-size 640")
        if not 0.0 <= args.confidence_threshold <= 1.0:
            raise ValueError("--confidence-threshold must be in [0, 1]")
        image, tensor = load_preprocessed_image(args.image, args.input_size)
        model_id = MODEL_IDS[args.model]
        pytorch = run_pytorch(model_id, args.revision, args.weight_file,
                              args.sha256, tensor, args.confidence_threshold)
        onnx = run_onnx(args.onnx, tensor, args.confidence_threshold)
        coreml = run_coreml(args.coreml, image, args.confidence_threshold)
        pytorch_onnx = assert_parity(
            "PyTorch", pytorch, "ONNX", onnx, args.iou_threshold,
            args.score_threshold, args.min_match_ratio, args.max_count_delta)
        pytorch_coreml = assert_parity(
            "PyTorch", pytorch, "CoreML", coreml, args.iou_threshold,
            args.score_threshold, args.min_match_ratio, args.max_count_delta)
        onnx_coreml = assert_parity(
            "ONNX", onnx, "CoreML", coreml, args.iou_threshold,
            args.score_threshold, args.min_match_ratio, args.max_count_delta)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"pytorch_detections={len(pytorch)}")
    print(f"onnx_detections={len(onnx)}")
    print(f"coreml_detections={len(coreml)}")
    print(f"pytorch_onnx_matches={len(pytorch_onnx)}")
    print(f"pytorch_coreml_matches={len(pytorch_coreml)}")
    print(f"onnx_coreml_matches={len(onnx_coreml)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
