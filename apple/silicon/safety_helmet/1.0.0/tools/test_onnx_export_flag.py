import torch
from modelscope.models import Model

model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
model = Model.from_pretrained(model_id)

model.eval()

# Dummy input [1, 3, 640, 640]
dummy_input = torch.randn(1, 3, 640, 640)

print("--- With onnx_export = False ---")
try:
    with torch.no_grad():
        out = model(dummy_input)
    print("Output types/shapes when onnx_export = False:")
    if isinstance(out, tuple):
        for i, val in enumerate(out):
            print(f"out[{i}] type: {type(val)}")
            if torch.is_tensor(val):
                print(f"shape: {val.shape}")
    else:
        print(type(out))
except Exception as e:
    print("Failed with onnx_export=False:", e)

print("\n--- With onnx_export = True ---")
model.onnx_export = True
try:
    with torch.no_grad():
        out = model(dummy_input)
    print("Output types/shapes when onnx_export = True:")
    if isinstance(out, tuple):
        for i, val in enumerate(out):
            print(f"out[{i}] type: {type(val)}")
            if torch.is_tensor(val):
                print(f"shape: {val.shape}")
            elif isinstance(val, tuple):
                print(f"sub-tuple lengths: {[v.shape if torch.is_tensor(v) else type(v) for v in val]}")
    else:
        print(type(out))
        if torch.is_tensor(out):
            print(f"shape: {out.shape}")
except Exception as e:
    print("Failed with onnx_export=True:", e)
