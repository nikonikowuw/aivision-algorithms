import torch
from modelscope.models import Model

model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
model = Model.from_pretrained(model_id)

print("model.onnx_export:", getattr(model, 'onnx_export', None))
# Let's inspect the code of model.forward or see what it returns
import inspect
try:
    print("\nForward signature:")
    print(inspect.signature(model.forward))
except Exception as e:
    print(e)

# Let's check how DamoYolo itself is structured. It has backbone, neck, head.
# Let's print the sub-modules
print("\nSub-modules:")
for name, module in model.named_children():
    print(name, type(module))

# Let's check what head returns
print("\nHead type:", type(model.head))
print("Head signature:")
print(inspect.signature(model.head.forward))
