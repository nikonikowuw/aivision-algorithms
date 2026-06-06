# 人脸识别算法包使用示例

本目录包含人脸识别算法包的使用示例代码。

## 示例列表

### 1. basic_usage.cpp (C++示例)

基本的C++使用示例，展示如何加载算法包、初始化、执行推理和销毁。

#### 编译

```bash
make
```

#### 运行

```bash
./basic_usage /path/to/face_recognition/1.0.0
```

### 2. basic_usage.py (Python示例)

基本的Python使用示例，展示如何使用ctypes调用算法包。

#### 运行

```bash
python3 basic_usage.py /path/to/face_recognition/1.0.0
```

## 使用流程

### 1. 准备算法包

确保算法包目录包含以下文件：

```
face_recognition/1.0.0/
├── algo_meta.yaml
├── nikoniko_detector.so
├── testimage.jpg
├── label_map.json
└── models/
    ├── insightface/
    │   ├── det_10g.onnx
    │   └── w600k_r50.onnx
    └── yolov8n.onnx
```

### 2. 加载算法包

```cpp
// C++
void* handle = dlopen("path/to/nikoniko_detector.so", RTLD_NOW | RTLD_LOCAL);
```

```python
# Python
import ctypes
lib = ctypes.CDLL("path/to/nikoniko_detector.so")
```

### 3. 初始化算法

```cpp
// C++
const char* config_json = "{\"package_dir\": \"path/to/package\", \"conf_thres\": 0.5}";
algo_handle_t algo_handle = detector_init(config_json);
```

```python
# Python
config = {"package_dir": "path/to/package", "conf_thres": 0.5}
config_json = json.dumps(config).encode('utf-8')
handle = lib.detector_init(config_json)
```

### 4. 执行推理

```cpp
// C++
hw_buffer_desc_t input;
input.width = 640;
input.height = 480;
// ... 设置其他参数

infer_result_t result;
int ret = detector_infer(algo_handle, &input, nullptr, &result);
```

```python
# Python
result = face_recognition.infer(image_data, width, height)
```

### 5. 处理结果

```cpp
// C++
if (ret == 0) {
    // 解析result.result_json
    // 释放结果
    detector_free_result(&result);
}
```

```python
# Python
if result["success"]:
    print(result["result"])
```

### 6. 销毁算法

```cpp
// C++
detector_destroy(algo_handle);
dlclose(handle);
```

```python
# Python
face_recognition.destroy()
```

## 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conf_thres` | number | 0.5 | 人体检测置信度阈值 |
| `iou_thres` | number | 0.45 | NMS IOU阈值 |
| `enable_tracker` | boolean | true | 是否启用ByteTracker人体追踪 |
| `face_conf_thres` | number | 0.6 | 人脸检测置信度阈值 |
| `face_quality_thres` | number | 0.3 | 人脸质量阈值 |
| `max_faces` | integer | 10 | 单帧最大人脸数 |

## 输出格式

```json
{
  "results": [
    {
      "category_code": 13001,
      "identity_id": "employee_001",
      "similarity": 0.923,
      "detect_confidence": 0.981,
      "bbox": {
        "x": 120,
        "y": 150,
        "w": 40,
        "h": 40
      },
      "landmarks": [
        { "x": 130, "y": 160 },
        { "x": 145, "y": 160 },
        { "x": 137, "y": 170 }
      ],
      "track_id": 1,
      "quality_score": 0.85
    }
  ],
  "infer_time_us": 12345,
  "frame_width": 640,
  "frame_height": 480
}
```

## 错误处理

### 常见错误

1. **加载动态库失败**
   - 检查文件路径是否正确
   - 检查文件权限
   - 检查依赖库是否完整

2. **初始化失败**
   - 检查配置参数是否正确
   - 检查模型文件是否存在
   - 检查模型文件是否完整

3. **推理失败**
   - 检查输入图像格式是否正确
   - 检查图像尺寸是否合理
   - 检查算法是否已初始化

### 错误码

- `0`: 成功
- `-1`: 通用错误
- `-2`: 参数错误
- `-3`: 内存错误
- `-4`: 模型错误

## 性能优化

### 1. 使用GPU

如果系统支持GPU，可以在配置中启用：

```json
{
  "use_gpu": true,
  "gpu_id": 0
}
```

### 2. 批量处理

对于多张图片，可以批量处理以提高效率：

```python
for image in images:
    result = face_recognition.infer(image, width, height)
    # 处理结果
```

### 3. 缓存结果

对于重复的人脸，可以缓存特征向量：

```python
# 缓存特征向量
embedding_cache = {}

def get_embedding(face_image):
    face_hash = hash(face_image.tobytes())
    if face_hash not in embedding_cache:
        embedding_cache[face_hash] = extract_embedding(face_image)
    return embedding_cache[face_hash]
```

## 注意事项

1. **内存管理**: 确保及时释放推理结果和算法资源
2. **线程安全**: 算法实例不是线程安全的，多线程使用需要加锁
3. **模型兼容性**: 确保模型文件与算法版本兼容
4. **图像格式**: 支持RGB格式的图像数据
5. **图像尺寸**: 建议使用合理的图像尺寸，避免过大或过小