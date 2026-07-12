import argparse
import sys
import subprocess
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="SCRFD Person Model Converter for Apple Silicon (CoreML), Rockchip (RKNN), and Huawei Ascend (OM)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="weights/scrfd_person_2.5g.onnx",
        help="Path to input ONNX model. Default: weights/scrfd_person_2.5g.onnx"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        required=True,
        choices=["coreml", "rknn", "om"],
        help="Target platform format. Options: coreml, rknn, om"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output converted model file/directory. If not provided, it auto-generates."
    )
    parser.add_argument(
        "--soc",
        type=str,
        default=None,
        help="Target SoC platform. Default for RKNN: rk3588, default for OM: Ascend310P3"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable NPU INT8 quantization (primarily for RKNN)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset path for quantization (text file containing image paths)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Model input image size (height width). Default: 640 640"
    )
    return parser.parse_args()

def convert_to_coreml(onnx_path, output_path, imgsz):
    """Converts SCRFD ONNX model to CoreML using onnx2torch and coremltools."""
    print(f"🔄 Converting SCRFD Person ONNX model {onnx_path} to CoreML (.mlpackage)...")
    try:
        import coremltools as ct
        import torch
        from onnx2torch import convert as onnx2torch_convert
    except ImportError as e:
        print(f"\n❌ Error: Missing dependencies: {e}", file=sys.stderr)
        print("💡 Please install coremltools and onnx2torch in your python environment:", file=sys.stderr)
        print("   pip install coremltools onnx2torch\n", file=sys.stderr)
        sys.exit(1)
        
    final_output = output_path if output_path else str(onnx_path.with_suffix(".mlpackage"))
    
    try:
        print("-> Loading ONNX model into PyTorch using onnx2torch...")
        torch_model = onnx2torch_convert(str(onnx_path))
        torch_model.eval()
        
        print(f"-> Tracing model with input shape (1, 3, {imgsz[0]}, {imgsz[1]})...")
        dummy_input = torch.randn(1, 3, imgsz[0], imgsz[1])
        traced_model = torch.jit.trace(torch_model, dummy_input)
        
        print("-> Converting to CoreML...")
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=dummy_input.shape)]
        )
        
        mlmodel.save(final_output)
        print(f"✅ CoreML conversion successful! Output: {final_output}")
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

def convert_to_rknn(onnx_path, output_path, soc, quantize, dataset_path):
    """Converts ONNX model to RKNN using rknn-toolkit2."""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("\n⚠️  Warning: rknn-toolkit2 (rknn.api) is not installed on this system.", file=sys.stderr)
        print("   RKNN conversions must typically run on an x86_64 Ubuntu PC (or Rockchip board).", file=sys.stderr)
        print("   Please install 'rknn-toolkit2' to run this conversion locally.", file=sys.stderr)
        print(f"   ONNX model ready at: {onnx_path}\n", file=sys.stderr)
        sys.exit(1)
        
    target_platform = soc if soc else "rk3588"
    final_output = output_path if output_path else str(onnx_path.with_suffix(".rknn"))
    
    print(f"🔄 Converting ONNX model {onnx_path} to RKNN ({target_platform})...")
    rknn = RKNN(verbose=True)
    
    # Pre-process configuration (BGR->RGB mean=127.5, std=127.5 for SCRFD)
    print("-> Configuring RKNN...")
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[127.5, 127.5, 127.5]],
        target_platform=target_platform
    )
    
    # Load ONNX
    print("-> Loading ONNX model...")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print("❌ Load ONNX failed!", file=sys.stderr)
        sys.exit(ret)
        
    # Build RKNN
    print("-> Building RKNN model...")
    do_quant = quantize
    if do_quant and not dataset_path:
        print("⚠️  Warning: Quantization is requested but no --dataset calibration file is provided.", file=sys.stderr)
        
    ret = rknn.build(do_quantization=do_quant, dataset=dataset_path)
    if ret != 0:
        print("❌ Build RKNN failed!", file=sys.stderr)
        sys.exit(ret)
        
    # Export RKNN
    print(f"-> Exporting to: {final_output} ...")
    ret = rknn.export_rknn(final_output)
    if ret != 0:
        print("❌ Export RKNN failed!", file=sys.stderr)
        sys.exit(ret)
        
    print(f"✅ RKNN conversion completed successfully! Output: {final_output}")

def convert_to_om(onnx_path, output_path, soc, imgsz):
    """Converts ONNX model to Ascend OM using Ascend atc tool."""
    target_soc = soc if soc else "Ascend310P3"
    output_prefix = Path(output_path).with_suffix("") if output_path else onnx_path.with_suffix("")
    
    # Construct Ascend ATC compilation command
    atc_cmd = [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_prefix}",
        f"--soc_version={target_soc}",
        f"--input_shape=input:1,3,{imgsz[0]},{imgsz[1]}"
    ]
    
    cmd_str = " ".join(atc_cmd)
    print("\n=========================================")
    print("Ascend OM ATC Compilation Command:")
    print(f"  {cmd_str}")
    print("=========================================\n")
    
    if shutil.which("atc"):
        print("⚡️ 'atc' command found locally. Executing compiler...")
        try:
            res = subprocess.run(atc_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print(res.stdout)
            print(f"✅ OM compilation completed successfully! Output: {output_prefix}.om")
        except subprocess.CalledProcessError as e:
            print(f"❌ ATC compilation failed with exit code {e.returncode}!", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(e.returncode)
    else:
        print("⚠️  Warning: Ascend ATC compiler ('atc' command) was not found in your PATH.")
        print("   This command is typically executed inside a Huawei Ascend CANN environment.")
        print("   Please run the command above on a machine with CANN Toolkit installed.")
        print(f"   ONNX source model ready at: {onnx_path}\n")

def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input ONNX model '{input_path}' does not exist.", file=sys.stderr)
        print("💡 Hint: Please run the export script to copy/verify the ONNX file first, or download it:", file=sys.stderr)
        print(f"   python python/export.py --download-onnx", file=sys.stderr)
        sys.exit(1)
        
    if args.target == "coreml":
        convert_to_coreml(input_path, args.output, args.imgsz)
    elif args.target == "rknn":
        convert_to_rknn(input_path, args.output, args.soc, args.quantize, args.dataset)
    elif args.target == "om":
        convert_to_om(input_path, args.output, args.soc, args.imgsz)

if __name__ == "__main__":
    main()
