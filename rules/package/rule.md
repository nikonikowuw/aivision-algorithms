# 算法包打包规范 (Algorithm Package Standards)

本文件定义智能检测系统（AIVisionInference）的 C++ 算法包打包规范。
为了保证平台（Go 控制面 + Asynq 后台任务）能正确解压、解析元数据并进行 C++ 推理自检，所有上传的算法包必须严格遵循本规范。

---

## 1. 算法包标准目录结构

算法包在打包为 `.zip` 时，其根目录（或剥离单一顶层目录后的根目录）必须包含以下文件和目录。严禁缺失必需文件，或将文件混杂在多层嵌套中。

```text
算法包根目录/
├── nikoniko_detector.so  # 必需：固定命名的算法入口动态库
├── algo_meta.yaml        # 必需：算法元数据及参数定义文件
├── label_map.json        # 必需：模型类别到平台 category_code 的映射文件
├── testimage.jpg         # 必需：用于平台自检推理的测试图片
├── README.md             # 必需：算法包的说明文档（包含版本变更等）
└── models/               # 推荐：模型权重文件目录（支持存放多个模型权重如 .onnx, .om, .rknn 等）
```

## 2. 打包与解压契约 (Zip Archive Rules)

* **压缩格式**：必须为标准 `.zip` 格式，不支持 `.rar`, `.tar.gz` 等其他格式。
* **顶层目录剥离 (Flat Root Directory)**：
  * 平台在解包时，会检查所有文件是否共享同一个顶层目录（例如：所有文件都在 `face_recognition_1.0.0/` 目录下）。
  * **如果是**：平台会自动剥离该顶层前缀，将内容平铺解压。
  * **如果否（混杂层级）**：例如一部分文件在根目录，一部分在其它文件夹，或者没有统一的前缀，平台将保留其原始结构。若因此导致 `nikoniko_detector.so` 等关键文件不在解压根目录下，**自检和加载将会失败**。
  * *推荐打包做法*：直接对包含上述必需文件的目录进行打包，或在包含这些文件的目录下使用 `zip -r output.zip ./*` 进行打包，以防多层嵌套干扰。

## 3. Makefile 自动化打包

算法包根目录必须提供 `Makefile`，并在其中包含 `make package` 目标，以实现一键校验和打包。

## 4. 发布前自检清单 (Pre-Release Checklist)

算法开发人员在向平台发布/上传算法包前，必须手动或通过脚本核对以下清单：

1. `[ ]` **文件完整性**：验证 `nikoniko_detector.so`、`algo_meta.yaml`、`label_map.json` 和 `testimage.jpg` 均存在于发布包中。
2. `[ ]` **ABI 符号校验**：使用 `nm -gU` (macOS) 或 `nm -D` (Linux) 校验 `nikoniko_detector.so` 中是否完整导出了 C ABI 所需的核心符号。
3. `[ ]` **依赖隔离**：检查 `.so` 是否没有未定义的第三方依赖符号。尽量采用静态链接，防止由于边缘节点缺少库文件导致 `dlopen` 失败。
4. `[ ]` **参数暴露安全**：核对 `algo_meta.yaml` 中的参数定义，确认没有包含模型文件具体绝对路径、设备 ID 等部署敏感配置。
5. `[ ]` **类别编码合规**：确认 `label_map.json` 中配置的 `category_code` 均为五位整数（`10000-99999`），且在算法推理结果中返回的 `category_code` 与之一致。
6. `[ ]` **本地推理自检**：在本地运行 `detector_self_test`，确认能够在不依赖视频流的情况下，仅加载 `testimage.jpg` 成功完成推理。
