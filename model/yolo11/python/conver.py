import argparse
import sys
import subprocess
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11 Model Converter for Apple Silicon (CoreML), Rockchip (RKNN), and Huawei Ascend (OM)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="weights/yolo11n.pt",
        help="Path to input model file (.pt or .onnx). Default: weights/yolo11n.pt"
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
        default=640,
        help="Model input image size. Default: 640"
    )
    
    return parser.parse_args()

def export_pt_to_onnx(pt_path, imgsz):
    """Exports PyTorch model to ONNX using Ultralytics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics library is required to convert .pt to .onnx. Please install it.", file=sys.stderr)
        sys.exit(1)
        
    print(f"🔄 Exporting PyTorch model {pt_path} to ONNX format...")
    try:
        model = YOLO(pt_path)
        onnx_path = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
        print(f"✅ ONNX export successful: {onnx_path}")
        return Path(onnx_path)
    except Exception as e:
        print(f"❌ Failed to export PyTorch model to ONNX: {e}", file=sys.stderr)
        sys.exit(1)

def convert_to_coreml(input_path, output_path, imgsz):
    """Converts PyTorch model to CoreML using Ultralytics."""
    if input_path.suffix != ".pt":
        print("Error: CoreML export directly from ONNX is not recommended via ultralytics. Please use a PyTorch (.pt) weights file.", file=sys.stderr)
        sys.exit(1)
        
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics library is required to export to CoreML. Please install it.", file=sys.stderr)
        sys.exit(1)
        
    print(f"🔄 Exporting PyTorch model {input_path} to CoreML (.mlpackage)...")
    try:
        model = YOLO(input_path)
        # Note: export will output in the same directory as input_path
        exported_path = model.export(format="coreml", imgsz=imgsz, half=True)
        exported_path = Path(exported_path)
        
        if output_path:
            out_p = Path(output_path)
            if out_p.exists():
                if out_p.is_dir():
                    shutil.rmtree(out_p)
                else:
                    out_p.unlink()
            shutil.move(str(exported_path), str(out_p))
            print(f"✅ CoreML model saved to: {out_p}")
        else:
            print(f"✅ CoreML model saved to: {exported_path}")
    except Exception as e:
        print(f"❌ CoreML export failed: {e}", file=sys.stderr)
        sys.exit(1)

def convert_to_rknn(input_path, output_path, soc, quantize, dataset_path, imgsz):
    """Converts ONNX model to RKNN using rknn-toolkit2."""
    onnx_path = input_path
    if input_path.suffix == ".pt":
        onnx_path = export_pt_to_onnx(input_path, imgsz)
        
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
    
    # Pre-process configuration (Standard normalizations for YOLO)
    print("-> Configuring RKNN...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
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
        print("⚠️  Warning: Quantization is requested but no --dataset calibration file is provided. Quantization will run without calibration dataset or fall back.", file=sys.stderr)
        
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

def convert_to_om(input_path, output_path, soc, imgsz):
    """Converts ONNX model to Ascend OM using Ascend atc tool."""
    onnx_path = input_path
    if input_path.suffix == ".pt":
        onnx_path = export_pt_to_onnx(input_path, imgsz)
        
    target_soc = soc if soc else "Ascend310P3"
    output_prefix = Path(output_path).with_suffix("") if output_path else onnx_path.with_suffix("")
    
    # Construct Ascend ATC compilation command
    # Framework = 5 represents ONNX
    atc_cmd = [
        "atc",
        f"--model={onnx_path}",
        "--framework=5",
        f"--output={output_prefix}",
        f"--soc_version={target_soc}",
        f"--input_shape=images:1,3,{imgsz},{imgsz}"
    ]
    
    cmd_str = " ".join(atc_cmd)
    print("\n=========================================")
    print("Ascend OM ATC Compilation Command:")
    print(f"  {cmd_str}")
    print("=========================================\n")
    
    # Check if 'atc' is available in local command line path
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
        print("   This command is typically executed inside a Huawei Ascend CANN environment/development machine.")
        print("   Please run the command above on a machine with CANN Toolkit installed.")
        print(f"   ONNX source model ready at: {onnx_path}\n")

def main():
    args = parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input model '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if args.target == "coreml":
        convert_to_coreml(input_path, args.output, args.imgsz)
    elif args.target == "rknn":
        convert_to_rknn(input_path, args.output, args.soc, args.quantize, args.dataset, args.imgsz)
    elif args.target == "om":
        convert_to_om(input_path, args.output, args.soc, args.imgsz)

if __name__ == "__main__":
    main()
