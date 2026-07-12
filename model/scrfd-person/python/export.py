import argparse
import sys
import shutil
import urllib.request
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="SCRFD Person Model Export & Sync Script")
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="weights/scrfd_person_2.5g.onnx",
        help="Path to SCRFD Person ONNX model. Default: weights/scrfd_person_2.5g.onnx"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output ONNX model. If not provided, it auto-generates."
    )
    parser.add_argument(
        "--download-onnx",
        action="store_true",
        help="Directly download the official pre-converted scrfd_person_2.5g.onnx model"
    )
    return parser.parse_args()

def download_file(url, output_path):
    print(f"📥 Downloading from {url} to {output_path}...")
    try:
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
    output_path = Path(args.output) if args.output else weights_path
    
    if args.download_onnx:
        url = "https://huggingface.co/public-data/insightface/resolve/main/models/scrfd_person_2.5g.onnx"
        success = download_file(url, output_path)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.", file=sys.stderr)
        print("💡 Hint: You can download the ONNX model directly by running:", file=sys.stderr)
        print(f"   python {sys.argv[0]} --download-onnx", file=sys.stderr)
        sys.exit(1)
        
    print(f"✅ ONNX model already exists and verified at: {weights_path}")
    if output_path != weights_path:
        print(f"🔄 Copying to target output: {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_path, output_path)
            print("✅ Copy successful!")
        except Exception as e:
            print(f"❌ Copy failed: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
