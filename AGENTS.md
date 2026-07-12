# 智能检测系统 C++ 算法包开发规范索引 (Algorithm AGENTS.md)

本文件定义 `algorithm/` 目录下 C++ Engine 算法包的开发专属规范索引。在 `algorithm/` 目录下执行任何任务（包括开发、构建、重构与优化算法包）前，**AI Agent 必须依次阅读并严格遵循以下专项规则**：

---

## 1. 核心架构与开发契约

* **项目物理架构与构建规范**：严格遵循 [.rules/oracle/rule.md] 规定的物理分层、单向依赖与 Target-based Modern CMake 构建标准。
* **算法开发通用规范与防御性编程**：严格遵循 [.rules/algo-dev/rule.md]，做好指针校验、异常安全隔离与 RAII 资源管理，严防进程崩溃。

## 2. 个个平台的开发规范

* **Apple Silicon (Mac M 系列)**: 严格遵循 [.rules/m-dev/rule.md] 中的开发规范。

## 2. 开发基本原则

1. **严守崩溃红线**：所有外部调用接口（`algo_init`/`algo_infer`/`algo_destroy`）必须包裹最外层 `try-catch` 块；任何脏输入、SDK 报错均不允许向上抛出异常或调用 `exit()` 等进程中断函数。
2. **坐标归一化**：推理结果 JSON 字符串中，检测框 `bbox` 必须基于**原始输入帧**归一化为 `[0.0, 1.0]` 的 `[x, y, w, h]` 相对坐标。
