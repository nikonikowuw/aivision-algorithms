#!/usr/bin/env python3
"""Convert testimage.jpg to raw BGR format for the e2e inference test."""
from PIL import Image
import struct

img = Image.open('testimage.jpg').convert('RGB')
# PIL loads as RGB; we need BGR so swap channels
r, g, b = img.split()
bgr = Image.merge('RGB', (b, g, r))
buf = bgr.tobytes()
with open('/tmp/testimage_raw.bgr', 'wb') as f:
    f.write(struct.pack('<II', img.width, img.height))
    f.write(buf)
print(f'  Converted testimage.jpg ({img.width}x{img.height}) to raw BGR')
