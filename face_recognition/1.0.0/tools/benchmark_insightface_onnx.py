#!/usr/bin/env python3
"""验证 InsightFace ONNX 模型只能通过 CoreML EP 初始化与推理。"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify InsightFace ONNX with CoreML EP only")
    parser.add_argument("models", nargs="*", default=[
        "models/insightface/2d106det.onnx",
        "models/insightface/w600k_r50.onnx",
    ])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def make_input(shape: Sequence[int]) -> np.ndarray:
    normalized = []
    for dim in shape:
        if dim is None or isinstance(dim, str):
            normalized.append(1)
        else:
            value = int(dim)
            normalized.append(1 if value <= 0 else value)
    return np.random.rand(*normalized).astype(np.float32)


def benchmark_model(model_path: Path, runs: int, warmup: int) -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise SystemExit(
            "无法导入 onnxruntime。建议使用隔离环境: "
            "python3 -m venv .venv-mac-ml && "
            ". .venv-mac-ml/bin/activate && "
            "python -m pip install 'numpy<2' onnxruntime-silicon"
        ) from exc

    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in available:
        raise SystemExit(f"CoreMLExecutionProvider 不可用，available={available}")

    session_options = ort.SessionOptions()
    session_options.log_severity_level = 2
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram",
            "MLComputeUnits": "ALL",
            "RequireStaticInputShapes": "0",
            "EnableOnSubgraphs": "1",
        })],
    )

    actual_providers = session.get_providers()
    inputs = {item.name: make_input(item.shape) for item in session.get_inputs()}
    for _ in range(warmup):
        session.run(None, inputs)

    elapsed_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, inputs)
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    mean = statistics.mean(elapsed_ms)
    p50 = statistics.median(elapsed_ms)
    p95 = sorted(elapsed_ms)[int(runs * 0.95) - 1]
    print(
        f"[OK] {model_path} providers={actual_providers} "
        f"mean={mean:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms"
    )


def main() -> None:
    args = parse_args()
    for raw_path in args.models:
        model_path = Path(raw_path)
        if not model_path.exists():
            print(f"[MISS] {model_path}")
            continue
        benchmark_model(model_path, args.runs, args.warmup)


if __name__ == "__main__":
    main()
