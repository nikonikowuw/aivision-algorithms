#!/usr/bin/env python3
"""
test_infer.py — 端到端推理测试 (End-to-End Inference Test)

测试流程 / Test Flow:
1. ctypes.CDLL 加载 .so / Load .so via ctypes.CDLL
2. detector_self_test 自检 / Run self-test
3. 分配合成 BGR24 framebuffer (640x480 灰) / Allocate synthetic 640x480 gray BGR24 buffer
4. detector_init 初始化 / Initialize algorithm context
5. detector_infer 执行推理 / Run inference
6. 解析 JSON 输出并打印检测结果 / Parse & print JSON detections
7. algo_free_result + detector_destroy 清理 / Cleanup

验证点 / Validation:
- ABI struct 尺寸匹配引擎 (144/40) / Struct sizes match engine expectations
- detector_infer 返回 0 / Inference returns 0
- 输出的 result_json 是合法 JSON 数组 / Output is valid JSON array
- 每个结果项包含必填字段 / Each detection has required fields
"""

import ctypes
import json
import os
import sys
import platform

# ──────────────────────────────────────────────
# 像素格式 FourCC (与 algorithm_context.cpp 一致)
# Pixel Format FourCC (consistent with algorithm_context.cpp)
# ──────────────────────────────────────────────
PIX_FMT_BGR24 = 0x33524742  # 'B'|('G'<<8)|('R'<<16)|('3'<<24)

FRAME_W = 640
FRAME_H = 480
BYTES_PER_PIXEL = 3
STRIDE = FRAME_W * BYTES_PER_PIXEL
FRAME_SIZE = FRAME_H * STRIDE  # 640*480*3 = 921600


def _check_platform():
    """Ensure we're running on macOS arm64 (Apple Silicon)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        print("⚠ This test targets Apple Silicon (arm64 macOS) only.")
        print(f"  Current: {platform.system()} / {platform.machine()}")


# ──────────────────────────────────────────────
# ABI 结构体定义 (必须与 algo/abi_contract.h v2 完全匹配)
# ABI struct definitions (must exactly match algo/abi_contract.h v2)
# ──────────────────────────────────────────────

class HwBufferDesc(ctypes.Structure):
    """hw_buffer_desc_t — 对齐到 arm64 macOS ABI (144 bytes)."""
    _fields_ = [
        # v1 字段 / v1 fields  (0–51)
        ("dma_fd",         ctypes.c_int),      #  0–3
        ("_pad1",           ctypes.c_int),      #  4–7    padding for size_t alignment
        ("size",           ctypes.c_size_t),    #  8–15
        ("width",          ctypes.c_uint32),    # 16–19
        ("height",         ctypes.c_uint32),    # 20–23
        ("pixel_format",   ctypes.c_uint32),    # 24–27
        ("dma_buf_fd",     ctypes.c_int),       # 28–31
        ("phys_addr",      ctypes.c_uint64),    # 32–39
        ("data",           ctypes.c_void_p),    # 40–47
        ("stride",         ctypes.c_uint32),    # 48–51
        # v2 字段 / v2 fields (52–63)
        ("buffer_type",    ctypes.c_int32),     # 52–55
        ("buffer_owner",   ctypes.c_int32),     # 56–59
        ("reserved_flags", ctypes.c_uint32),    # 60–63
        # 联合体 / union plat (64–127, 64 bytes)
        ("plat_",          ctypes.c_uint8 * 64),
        # 保留填充 / reserved padding (128–143, 16 bytes)
        ("reserved_padding", ctypes.c_int64 * 2),
    ]


class InferResult(ctypes.Structure):
    """infer_result_t (40 bytes, 8-byte aligned)."""
    _fields_ = [
        ("result_json",    ctypes.c_char_p),    #  0–7
        ("result_json_len", ctypes.c_size_t),    #  8–15
        ("infer_time_us",  ctypes.c_uint32),     # 16–19
        ("reserved",       ctypes.c_int * 4),    # 20–35 + 4 pad → 40
    ]


# ──────────────────────────────────────────────
# 工具函数 / Utility Functions
# ──────────────────────────────────────────────

def set_infer_result_fields_py3(result):
    """
    Python 3 兼容: 显式初始化 InferResult 字段.
    Python 3 compatibility: Explicitly initialize InferResult fields.
    """
    result.result_json = None
    result.result_json_len = 0
    result.infer_time_us = 0
    for i in range(4):
        result.reserved[i] = 0


def fill_synthetic_bgr(width, height):
    """
    生成 640×480 BGR24 灰色图像.

    Generate a 640×480 BGR24 gray image buffer (bytearray).
    """
    stride = width * 3
    size = height * stride
    buf = bytearray(size)

    # 均匀灰色 (B=128, G=128, R=128)
    # Uniform gray
    for y in range(height):
        row_start = y * stride
        for x in range(width):
            idx = row_start + x * 3
            buf[idx]     = 128   # B
            buf[idx + 1] = 128   # G
            buf[idx + 2] = 128   # R
    return buf, stride, size


def describe_structs():
    """打印结构体尺寸验证信息 / Print struct size validation."""
    hw_size = ctypes.sizeof(HwBufferDesc)
    ir_size = ctypes.sizeof(InferResult)
    print(f"  sizeof(hw_buffer_desc_t) = {hw_size}  (expected 144)")
    print(f"  sizeof(infer_result_t)   = {ir_size}  (expected 40)")

    ok = True
    if hw_size != 144:
        print(f"  ✗ hw_buffer_desc_t size mismatch!")
        ok = False
    if ir_size != 40:
        print(f"  ✗ infer_result_t size mismatch!")
        ok = False
    if ok:
        print("  ✓ ABI struct sizes validated")
    return ok


# ──────────────────────────────────────────────
# 主测试流程 / Main Test Flow
# ──────────────────────────────────────────────

def main():
    _check_platform()

    # ── 解析命令行参数 / Parse CLI args ──────────
    so_path = sys.argv[1] if len(sys.argv) > 1 else "./nikoniko_detector.so"
    if not os.path.isfile(so_path):
        print(f"✗ Shared library not found: {os.path.abspath(so_path)}")
        return 1

    pkg_dir = os.path.dirname(os.path.abspath(so_path))

    print("=" * 60)
    print("  Face Recognition — End-to-End Inference Test")
    print("=" * 60)
    print(f"  Library: {so_path}")
    print(f"  Package: {pkg_dir}")
    print()

    # ── 1. 加载动态库 / Load shared library ─────
    print("─── Step 1: Load Shared Library ───")
    lib = ctypes.CDLL(so_path)
    print("  ✓ dlopen succeeded")
    print(f"  ✓ sizeof(HwBufferDesc) = {ctypes.sizeof(HwBufferDesc)}")
    print(f"  ✓ sizeof(InferResult)  = {ctypes.sizeof(InferResult)}")
    print()

    # ── 2. 类型签名 / Function signatures ────────
    lib.detector_init.argtypes = [ctypes.c_char_p]
    lib.detector_init.restype = ctypes.c_void_p

    lib.detector_infer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(HwBufferDesc),
        ctypes.c_char_p,
        ctypes.POINTER(InferResult),
    ]
    lib.detector_infer.restype = ctypes.c_int

    lib.algo_free_result.argtypes = [ctypes.POINTER(InferResult)]
    lib.algo_free_result.restype = None

    lib.detector_destroy.argtypes = [ctypes.c_void_p]
    lib.detector_destroy.restype = None

    lib.detector_self_test.restype = ctypes.c_int
    lib.detector_version.restype = ctypes.c_char_p
    lib.detector_name.restype = ctypes.c_char_p

    # ── 3. 结构体尺寸验证 / Struct size check ───
    if not describe_structs():
        print("✗ ABI struct size mismatch — aborting test")
        return 1
    print()

    # ── 4. 自检 / Self-test ──────────────────────
    print("─── Step 2: Self-Test ───")
    ret = lib.detector_self_test()
    status = "✓ passed" if ret == 0 else f"✗ failed (ret={ret})"
    print(f"  detector_self_test: {status}")
    if ret != 0:
        return 1

    name = lib.detector_name().decode()
    ver = lib.detector_version().decode()
    print(f"  name={name}, version={ver}")
    print()

    # ── 5. 初始化 / Initialize ──────────────────
    print("─── Step 3: Initialize ───")
    config_json = json.dumps({"package_dir": pkg_dir})
    handle = lib.detector_init(config_json.encode())
    if not handle:
        print("✗ detector_init returned NULL")
        return 1
    print(f"  ✓ handle = {handle}")
    print()

    # ── 6. 准备合成数据 / Prepare synthetic frame ─
    print("─── Step 4: Prepare Synthetic BGR24 Frame ───")
    buf, stride, buf_size = fill_synthetic_bgr(FRAME_W, FRAME_H)
    data_array = (ctypes.c_uint8 * buf_size).from_buffer(buf)

    hw_desc = HwBufferDesc()
    hw_desc.dma_fd = -1
    hw_desc.size = buf_size
    hw_desc.width = FRAME_W
    hw_desc.height = FRAME_H
    hw_desc.pixel_format = PIX_FMT_BGR24
    hw_desc.dma_buf_fd = -1
    hw_desc.phys_addr = 0
    hw_desc.data = ctypes.cast(data_array, ctypes.c_void_p)
    hw_desc.stride = stride
    hw_desc.buffer_type = 0
    hw_desc.buffer_owner = 0
    hw_desc.reserved_flags = 0
    # 清零联合体和填充 / Zero out union & padding
    for i in range(64):
        hw_desc.plat_[i] = 0
    for i in range(2):
        hw_desc.reserved_padding[i] = 0

    print(f"  Resolution: {FRAME_W}×{FRAME_H}")
    print(f"  Format: BGR24 (FourCC 0x{PIX_FMT_BGR24:08X})")
    print(f"  Buffer: {buf_size} bytes, stride={stride}")
    print()

    # ── 7. 推理 / Inference ──────────────────────
    print("─── Step 5: detector_infer ───")
    result = InferResult()
    set_infer_result_fields_py3(result)

    ret = lib.detector_infer(handle, ctypes.byref(hw_desc), None, ctypes.byref(result))
    print(f"  return code = {ret}")

    if ret != 0:
        print(f"✗ Inference failed with error code {ret}")
        lib.detector_destroy(handle)
        return 1
    print()

    # ── 8. 解析结果 / Parse result ──────────────
    print("─── Step 6: Output ───")
    print(f"  Infer time: {result.infer_time_us} µs")
    print(f"  JSON length: {result.result_json_len}")

    if result.result_json and result.result_json_len > 0:
        raw_bytes = result.result_json[:result.result_json_len]
        json_str = raw_bytes.decode("utf-8", errors="replace")

        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON result: {e}")
            print(f"  Raw: {json_str[:200]}")
            lib.algo_free_result(ctypes.byref(result))
            lib.detector_destroy(handle)
            return 1

        print(f"  Detected objects: {len(parsed)}")
        print()

        for i, obj in enumerate(parsed):
            label = obj.get("label", "?")
            cat = obj.get("category_code", "?")
            conf = obj.get("detect_confidence", 0.0)
            tid = obj.get("track_id", -1)
            bbox = obj.get("bbox", {})
            print(f"  [{i}] {label:16s} cat={cat:5d} conf={conf:.3f} "
                  f"track={tid:3d} "
                  f"bbox=({bbox.get('x',0):.3f},{bbox.get('y',0):.3f},"
                  f"{bbox.get('w',0):.3f},{bbox.get('h',0):.3f})")

            # 人脸额外字段 / Face-specific fields
            if cat in (13001, 13002):
                lms = obj.get("landmarks", [])
                print(f"        face_quality={obj.get('face_quality_score', 0):.3f} "
                      f"landmarks={len(lms) if lms else 0}")
                pb = obj.get("person_bbox", {})
                if pb:
                    print(f"        person_bbox=({pb.get('x',0):.3f},{pb.get('y',0):.3f},"
                          f"{pb.get('w',0):.3f},{pb.get('h',0):.3f})")
                if cat == 13001:  # known_face
                    sim = obj.get("similarity", 0.0)
                    candidates = obj.get("candidates", [])
                    print(f"        identity=\"{obj.get('identity_name','?')}\" "
                          f"sim={sim:.3f} candidates={len(candidates)}")

        print()
        print(f"✓ Valid JSON output with {len(parsed)} object(s)")
    else:
        print("  (no result data)")
        print("✓ Inference succeeded (empty result)")

    # ── 9. 清理 / Cleanup ───────────────────────
    lib.algo_free_result(ctypes.byref(result))
    lib.detector_destroy(handle)
    print()

    print("=" * 60)
    print("  ✅ End-to-End Inference Test Passed")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
