import argparse
import sys
import urllib.request
from pathlib import Path
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="SCRFD Model Export Script")
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="weights/scrfd_500m.pth",
        help="Path to SCRFD PyTorch model weights (.pth). Default: weights/scrfd_500m.pth"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to mmdet config file (e.g. configs/scrfd/scrfd_500m.py)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output ONNX model. If not provided, it auto-generates."
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[640, 640],
        help="Input image shape as height width. Default: 640 640"
    )
    parser.add_argument(
        "--download-onnx",
        action="store_true",
        help="Directly download the official pre-converted scrfd_500m.onnx model and skip export"
    )
    return parser.parse_args()

def download_file(url, output_path):
    print(f"📥 Downloading from {url} to {output_path}...")
    try:
        # User-agent bypass for standard download
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, output_path)
        print("✅ Download successful!")
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}", file=sys.stderr)
        return False

def main():
    args = parse_args()
    
    weights_path = Path(args.weights)
    output_path = Path(args.output) if args.output else weights_path.with_suffix(".onnx")
    
    if args.download_onnx:
        # Standard SCRFD 500M ONNX URL from Hugging Face
        url = "https://huggingface.co/ceyxprime/scrfd_640_batched/resolve/main/det_500m_fixed.onnx"
        success = download_file(url, output_path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.", file=sys.stderr)
        print("💡 Hint: You can download the pre-converted ONNX model directly by running:", file=sys.stderr)
        print(f"   python {sys.argv[0]} --download-onnx", file=sys.stderr)
        sys.exit(1)
        
    try:
        from mmdet.apis import init_detector
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector
    except ImportError:
        print("\n❌ Error: mmdet or mmcv is not installed in the python environment.", file=sys.stderr)
        print("💡 SCRFD PyTorch checkpoints require OpenMMLab libraries (mmdet and mmcv) to export.", file=sys.stderr)
        print("   If you do not want to install mmdet, you can download the official pre-converted ONNX model directly:", file=sys.stderr)
        print(f"   python {Path(__file__).name} --download-onnx\n", file=sys.stderr)
        sys.exit(1)
        
    if not args.config:
        print("❌ Error: Exporting from .pth requires the model config file. Please specify --config.", file=sys.stderr)
        print("   Example: configs/scrfd/scrfd_500m.py in the official insightface repo.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading SCRFD model configuration from: {args.config}")
    cfg = Config.fromfile(args.config)
    
    print(f"Building model and loading weights from: {weights_path}")
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, str(weights_path), map_location='cpu')
    model.cpu()
    model.eval()
    
    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, args.shape[0], args.shape[1])
    else:
        print("❌ Error: Invalid input shape. Must be 1 or 2 values.", file=sys.stderr)
        sys.exit(1)
        
    dummy_input = torch.randn(*input_shape)
    
    print(f"🔄 Exporting to ONNX: {output_path} with input shape {input_shape}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=['scores', 'bboxes', 'kps'] if 'kps' in cfg.model.bbox_head.type.lower() else ['scores', 'bboxes'],
            opset_version=11,
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print(f"✅ ONNX export successful! Saved to: {output_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
