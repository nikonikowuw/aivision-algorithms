# Models

fall_detection 1.0.0 是规则型摔倒检测算法，不依赖独立 ONNX/TensorRT 模型。
算法优先消费上游人体检测算法输出的 `context_json`，根据人体框宽高比与轨迹下落趋势判定摔倒事件。
