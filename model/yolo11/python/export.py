import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 Model Export Script")
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="yolo11n.pt",
        help="Path to PyTorch model weights (.pt file). Default: yolo11n.pt"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=[
            "onnx", "torchscript", "openvino", "engine", "coreml",
            "saved_model", "pb", "tflite", "edgetpu", "tfjs", "paddle", "ncnn"
        ],
        help="Export format. Default: onnx"
    )
    parser.add_argument(
        "--imgsz", "--img-size",
        type=int,
        nargs="+",
        default=[640],
        help="Image size as a single integer (e.g. 640) or two integers (height width). Default: 640"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16) for export"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes for ONNX/TensorRT"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (e.g. 17). Default: None (use ultralytics default)"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 quantization"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Batch size for export. Default: 1"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for export (e.g. cpu, cuda:0). Default: None"
    )

    args = parser.parse_args()

    # Process imgsz: if single value, convert to int, otherwise keep as list/tuple
    if len(args.imgsz) == 1:
        args.imgsz = args.imgsz[0]
    elif len(args.imgsz) == 2:
        args.imgsz = tuple(args.imgsz)
    else:
        parser.error("imgsz must be 1 or 2 values (e.g. --imgsz 640 or --imgsz 640 480)")

    return args

def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting model to format '{args.format}'...")
    try:
        # Call ultralytics export
        exported_path = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            opset=args.opset,
            int8=args.int8,
            batch=args.batch,
            device=args.device
        )
        print(f"Successfully exported! Output path: {exported_path}")
    except Exception as e:
        print(f"Error exporting model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

