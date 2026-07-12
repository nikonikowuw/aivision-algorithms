import argparse
import sys
from pathlib import Path
import torch

# Add current directory to path so python can import python.net
sys.path.append(str(Path(__file__).parent.parent))
from python.net import build_model

class BackboneWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, x):
        output, norm = self.base_model(x)
        return output

def parse_args():
    parser = argparse.ArgumentParser(description="AdaFace Model Export Script")
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default="weights/adaface_ir101_webface4m.ckpt",
        help="Path to PyTorch model weights (.ckpt or .pt file). Default: weights/adaface_ir101_webface4m.ckpt"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="onnx",
        choices=["onnx", "coreml"],
        help="Export format. Default: onnx"
    )
    parser.add_argument(
        "--imgsz", "--img-size",
        type=int,
        default=112,
        help="Input image size (assumed square). Default: 112"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version. Default: 14"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for export (cpu or cuda). Default: cpu"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output file/directory. If not provided, it auto-generates."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading AdaFace IR-101 model from: {weights_path}")
    try:
        model = build_model("ir_101")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[6:] if k.startswith("model.") else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(args.device)
        model.eval()
        print("✅ Weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Wrap model to output only the feature embedding tensor (excluding L2 norm)
    export_model = BackboneWrapper(model)
    export_model.eval()
    dummy_input = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    
    if args.format == "onnx":
        output_path = Path(args.output) if args.output else weights_path.with_suffix(".onnx")
        print(f"🔄 Exporting to ONNX: {output_path} ...")
        try:
            torch.onnx.export(
                export_model,
                dummy_input,
                str(output_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                opset_version=args.opset
            )
            
            # Post-process to consolidate external data weights into a single ONNX file
            print("-> Consolidating external data weights back into the ONNX file...")
            import onnx
            import os
            from onnx.external_data_helper import convert_model_from_external_data
            
            model = onnx.load(str(output_path))
            convert_model_from_external_data(model)
            onnx.save(model, str(output_path))
            
            # Clean up external data file if created
            data_file = output_path.with_suffix(output_path.suffix + ".data")
            if data_file.exists():
                os.remove(data_file)
                print(f"🧹 Cleaned up temporary external data file: {data_file}")
                
            print(f"✅ ONNX export successful! Output path: {output_path}")
        except Exception as e:
            print(f"❌ ONNX export failed: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.format == "coreml":
        output_path = Path(args.output) if args.output else weights_path.with_suffix(".mlpackage")
        print(f"🔄 Exporting to CoreML: {output_path} ...")
        try:
            import coremltools as ct
            
            # Trace the model first
            traced_model = torch.jit.trace(export_model, dummy_input)
            
            # Convert using coremltools
            mlmodel = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="input", shape=dummy_input.shape)],
                outputs=[ct.TensorType(name="output")]
            )
            
            # Save the model
            mlmodel.save(str(output_path))
            print(f"✅ CoreML export successful! Output path: {output_path}")
        except Exception as e:
            print(f"❌ CoreML export failed: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
