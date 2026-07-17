import torch
from modelscope.models import Model

model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
model = Model.from_pretrained(model_id)

model.eval()
model.onnx_export = True

dummy_input = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out = model(dummy_input)

print("Output shape:", out.shape)
print("First anchor output channel values:")
print(out[0, 0, :].tolist())

print("\nMean values across all 8400 anchors:")
print("x1/y1/x2/y2/cls0/cls1 mean:")
print(out.mean(dim=1)[0].tolist())

print("\nMin values across all 8400 anchors:")
print(out.min(dim=1)[0][0].tolist())

print("\nMax values across all 8400 anchors:")
print(out.max(dim=1)[0][0].tolist())
