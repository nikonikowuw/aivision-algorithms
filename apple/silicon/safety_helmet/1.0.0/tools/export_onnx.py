#!/usr/bin/env python3
"""Export DAMO-YOLO Safety Helmet model from ModelScope to ONNX format."""

import argparse
import os
import sys
import torch

def download_and_export(model_id, output_path, opset_version):
    """Download model from ModelScope and export to ONNX."""
    print(f"Loading ModelScope model '{model_id}'...")
    from modelscope.models import Model
    
    # Force loading of model
    model = Model.from_pretrained(model_id)
    model.eval()
    
    # Enable ONNX export flag
    model.onnx_export = True
    print("Enabled model.onnx_export = True")

    # Create dummy input [1, 3, 640, 640]
    dummy_input = torch.randn(1, 3, 640, 640)

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Exporting model to ONNX format at: {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        opset_version=opset_version,
        input_names=['images'],
        output_names=['outputs']
    )
    print("ONNX export completed.")

def verify_onnx(onnx_path):
    """Verify exported ONNX model using ONNXRuntime."""
    print("Verifying ONNX model with ONNXRuntime...")
    import onnxruntime as ort
    import numpy as np

    session_opts = ort.SessionOptions()
    session = ort.InferenceSession(onnx_path, session_opts, providers=['CPUExecutionProvider'])

    # Get inputs and outputs details
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print("\nONNX Model Info:")
    for idx, inp in enumerate(inputs):
        print(f"Input {idx}: name='{inp.name}', shape={inp.shape}, type={inp.type}")
    for idx, out in enumerate(outputs):
        print(f"Output {idx}: name='{out.name}', shape={out.shape}, type={out.type}")

    # Run single inference
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    res = session.run([out.name for out in outputs], {inputs[0].name: dummy_input})
    print(f"\nInference test successful. Output shape: {res[0].shape}")

def main():
    parser = argparse.ArgumentParser(description='Export DAMO-YOLO Safety Helmet to ONNX')
    parser.add_argument('--output', default='weights/damoyolo_safety_helmet.onnx',
                        help='Output path for the exported ONNX model')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11)')
    args = parser.parse_args()

    model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
    try:
        download_and_export(model_id, args.output, args.opset)
        verify_onnx(args.output)
        print("\n\x1b[32mSuccess: Model exported and verified successfully!\x1b[0m")
    except Exception as e:
        print(f"\n\x1b[31mError: ONNX export failed: {e}\x1b[0m")
        sys.exit(1)

if __name__ == '__main__':
    main()
