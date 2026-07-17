import os
import sys
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'iic/cv_tinynas_object-detection_damoyolo_safety-helmet'
print("Loading model from pretrained...")
model = Model.from_pretrained(model_id)
print("Model loaded.")
print("Type of model:", type(model))

# Print public attributes and methods
print("\nAttributes/methods:")
print([name for name in dir(model) if not name.startswith('_')])

# Let's inspect the checkpoint path or model_dir
model_dir = getattr(model, 'model_dir', None)
print("\nmodel_dir:", model_dir)
if model_dir and os.path.exists(model_dir):
    print("Files in model_dir:")
    print(os.listdir(model_dir))
