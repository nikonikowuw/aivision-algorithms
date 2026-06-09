#!/usr/bin/env python3
"""导出 YOLO 检测模型为 CoreML 格式。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model to CoreML for Apple Silicon")
    parser.add_argument("--person", default="models/yolo11n.pt", help="YOLO 人体检测 .pt 模型路径；缺失时自动下载 yolo11n.pt")
    parser.add_argument("--face", default="", help="可选：YOLO 人脸检测 .pt 模型路径")
    parser.add_argument("--imgsz", type=int, default=640, help="导出输入尺寸")
    parser.add_argument("--half", action="store_true", help="使用 FP16 CoreML 导出")
    return parser.parse_args()


def export_coreml(model_path: str, imgsz: int, half: bool) -> Path:
    from ultralytics import YOLO

    source = Path(model_path)
    if source.exists() and source.suffix != ".pt":
        raise ValueError(f"CoreML 导出建议使用 Ultralytics .pt 模型，当前是: {source}")

    source.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        model = YOLO(str(source))
    elif source.name == "yolo11n.pt":
        print(f"[INFO] {source} 不存在，使用 Ultralytics 自动下载 yolo11n.pt")
        model = YOLO("yolo11n.pt")
        downloaded = Path("yolo11n.pt")
        if downloaded.exists() and downloaded != source:
            shutil.move(str(downloaded), str(source))
            model = YOLO(str(source))
    else:
        raise FileNotFoundError(f"模型不存在: {source}")

    exported = Path(model.export(format="coreml", imgsz=imgsz, half=half))
    target = source.with_name("yolov11n.mlpackage")
    if exported != target:
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(exported), str(target))

    legacy = source.with_suffix(".mlpackage")
    if legacy != target and legacy.exists():
        shutil.rmtree(legacy) if legacy.is_dir() else legacy.unlink()

    return target


def main() -> None:
    args = parse_args()
    targets = [("person", args.person)]
    if args.face:
        targets.append(("face", args.face))

    for name, model_path in targets:
        try:
            exported = export_coreml(model_path, args.imgsz, args.half)
            print(f"[OK] {name}: {exported}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            raise


if __name__ == "__main__":
    main()
