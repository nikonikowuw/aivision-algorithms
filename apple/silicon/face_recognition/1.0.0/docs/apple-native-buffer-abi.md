# Apple Native Buffer ABI Roadmap

## 1. Scope

The current `hw_buffer_desc_t` exposes one host pointer and one stride. It cannot safely express a retained `CVPixelBufferRef`/`IOSurfaceRef`, independent NV12 plane layouts, native-object ownership, or producer/consumer synchronization.

Until Engine implements this contract, BGR24 may be wrapped as a non-owning view, but NV12 requires a stride-aware CPU conversion. The package must not advertise `zero_copy_input`.

## 2. Proposed Signature And Payload

Keep the existing ABI function signature:

```c
int detector_infer(algo_handle_t handle,
                   const hw_buffer_desc_t *input,
                   const char *context_json,
                   infer_result_t *result);
```

Define a versioned payload inside `hw_buffer_desc_t.plat[80]`, or introduce a new ABI version if the final layout does not fit:

```c
typedef enum {
    ALGO_APPLE_BUFFER_NONE = 0,
    ALGO_APPLE_BUFFER_CVPIXELBUFFER = 1,
    ALGO_APPLE_BUFFER_IOSURFACE = 2,
} algo_apple_buffer_kind_t;

typedef struct {
    uint16_t abi_version;       // Must be 1.
    uint16_t struct_size;       // sizeof(algo_apple_buffer_v1_t).
    uint32_t buffer_kind;       // algo_apple_buffer_kind_t.
    uint32_t pixel_format;      // CoreVideo/IOSurface FourCC.
    uint32_t plane_count;       // 1 for BGRA, 2 for NV12.
    uint64_t native_handle;     // Borrowed retained native object.
    uint32_t plane_stride[2];
    uint32_t plane_offset[2];
    uint64_t synchronization;   // Reserved synchronization token.
} algo_apple_buffer_v1_t;
```

Both Engine and algorithm builds must assert the payload size, alignment, and field offsets.

## 3. Ownership And Data Contract

- Engine owns the native object and keeps it valid for the complete synchronous `detector_infer` call.
- Engine retains a CoreFoundation object before the call and releases it after return.
- The algorithm borrows the handle and must not release it or store it beyond the call.
- Runtime FourCC, plane count, strides, and offsets must match the descriptor.
- NV12 exposes two planes. Never derive UV from `width * height`.
- Compatible CoreML models receive `CVPixelBuffer` as an image feature input.
- Custom preprocessing creates `CVMetalTextureCache` views of the same buffer and fuses resize, color conversion, and normalization.
- Packed BGR24 through `data/size/stride` remains the compatibility path.

Target flow:

```text
Decoder CVPixelBuffer
  -> Engine platform payload
  -> validated borrowed native handle
  -> CoreML image input or CVMetalTextureCache
  -> fused preprocessing
  -> inference
```

## 4. Validation And Errors

| Condition | Required behavior |
|---|---|
| Unknown ABI version or short payload | Return invalid-input error |
| Null or stale native handle | Return invalid-input error |
| NV12 with a plane count other than two | Return invalid-input error |
| Descriptor FourCC differs from native metadata | Return invalid-input error and log both values |
| Unsupported native format | Use CPU fallback only when explicitly permitted; otherwise return unsupported-format error |
| Missing synchronization readiness | Wait or return retry/busy according to the final synchronization contract |
| CoreML model has no image input | Use a fused Metal-to-tensor path; do not claim direct CoreML zero-copy |

No Objective-C or C++ exception may cross the C ABI boundary.

## 5. Cases

- Good: Engine passes a retained bi-planar NV12 `CVPixelBuffer`; the algorithm creates CoreML or Metal views without a full-frame CPU copy.
- Base: Engine passes packed BGR24 through `data`, `size`, and `stride`; the algorithm wraps it without copying.
- Bad: Engine passes an unowned pointer, or the algorithm assumes the UV plane starts at `width * height`.

## 6. Required Tests

- ABI layout assertions in the Engine language and C++.
- Retain/release lifetime test with exactly one release after inference.
- BGRA and bi-planar NV12 tests with padded and unequal plane strides.
- Negative tests for wrong version, size, plane count, FourCC, and stale handles.
- Pixel-equivalence comparison against the reference CPU preprocessing path.
- Instruments copy audit proving there is no full-frame CPU copy before model input.
- 1080p and 4K p50/p95 latency, CPU, bandwidth, ANE/GPU, and energy measurements.
- Packaging check that rejects `zero_copy_input` unless the native-buffer integration suite passes.

## 7. Wrong Vs Correct

Wrong:

```cpp
const uint8_t* uv = input->data + input->width * input->height;
```

Correct:

```cpp
const auto* apple = DecodeAppleBufferV1(input->plat, sizeof(input->plat));
CVPixelBufferRef pixel_buffer = BorrowValidatedPixelBuffer(*apple);
return coreml_backend.Run(pixel_buffer, outputs);
```

The correct path validates the versioned payload, native metadata, and ownership before inference.
