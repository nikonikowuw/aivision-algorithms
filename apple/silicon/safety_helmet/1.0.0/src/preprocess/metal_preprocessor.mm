/**
 * @file metal_preprocessor.mm
 * @brief Metal GPU preprocessing — ObjC++ implementation.
 *
 * ## Architecture
 *
 * Embeds two Metal Shading Language (MSL) compute kernels as a C string,
 * compiles them at construction time into MTLComputePipelineState objects,
 * and dispatches them per frame.
 *
 *   Kernel 1 — letterbox_from_buffer:
 *     Reads raw 8-bit BGR pixels from a MTLBuffer (uploaded from cv::Mat).
 *     Performs manual bilinear interpolation, BGR→RGB channel swap, and
 *     float32 normalisation. Used by Process().
 *
 *   Kernel 2 — letterbox_from_texture:
 *     Reads from a MTLTexture obtained via CVMetalTextureCache from a
 *     borrowed CVPixelBufferRef. Uses hardware-accelerated bilinear sampling
 *     with clamp-to-edge addressing. Used by ProcessFromPixelBuffer().
 *
 * ## Zero-copy contract
 *
 * ProcessFromPixelBuffer receives a *borrowed* CVPixelBufferRef from the
 * Engine. The algorithm must NEVER retain, release, or store it beyond the
 * synchronous detector_infer call. CVMetalTextureCacheCreateTextureFromImage
 * creates a temporary Metal texture view over the existing IOSurface backing
 * — no copy occurs. The texture is released before returning.
 *
 * ## Error handling
 *
 * All failures (no GPU, shader compilation error, dispatch error) are
 * reported via ALGO_LOG_ERROR and the method returns false. The caller
 * (SafetyHelmetPipeline) handles fallback to CPU preprocessing.
 */
#import "metal_preprocessor.h"
#import "common/logger.h"
#import <Metal/Metal.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Embedded MSL source — compiled at runtime via -[MTLDevice newLibraryWithSource:].
// Two compute kernels share the same letterbox parameter convention:
//   output[0*area + idx] = R, output[1*area + idx] = G, output[2*area + idx] = B
//   where idx = y * target + x, area = target * target.
// ---------------------------------------------------------------------------
static const char* kMetalShaderSource = R"metal(
#include <metal_stdlib>
using namespace metal;

/**
 * Bilinear letterbox preprocessing from a raw BGR uchar buffer.
 *
 * Reads 8-bit BGR triplets via manual bilinear interpolation (4-tap),
 * applies letterbox padding (gray=114/255), swaps BGR→RGB, and normalises
 * to [0,1]. Output layout is channel-first float32: R plane, G plane, B plane.
 *
 * Why manual interpolation instead of texture hardware?
 *   Avoids forcing the caller to pad BGR→BGRA for MTLTexture compatibility,
 *   saving one full-frame copy. The kernel's arithmetic is lightweight and
 *   memory-bandwidth bound, so the manual math penalty is negligible.
 */
kernel void letterbox_from_buffer(
    device const uchar* input   [[buffer(0)]],  // BGR pixels, row-major, 3 bytes/pixel
    device float*       output  [[buffer(1)]],  // CHW float tensor, 3*target*target
    constant uint&      input_w [[buffer(2)]],  // source width in pixels
    constant uint&      input_h [[buffer(3)]],  // source height in pixels
    constant uint&      target  [[buffer(4)]],  // output square side (e.g. 640)
    constant float&     scale   [[buffer(5)]],  // min(target/w, target/h)
    constant uint&      pad_left [[buffer(6)]], // integer left border used by CPU path
    constant uint&      pad_top  [[buffer(7)]], // integer top border used by CPU path
    constant uint&      scaled_w [[buffer(8)]], // rounded resized width
    constant uint&      scaled_h [[buffer(9)]], // rounded resized height
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= target || gid.y >= target) return;

    float r, g, b;

    // Match OpenCV's asymmetric odd padding exactly: the extra pixel belongs
    // to the bottom or right border, so content starts at integer left/top.
    if (gid.x < pad_left || gid.x >= pad_left + scaled_w ||
        gid.y < pad_top || gid.y >= pad_top + scaled_h) {
        r = g = b = 114.0f / 255.0f;
    } else {
        // Match OpenCV INTER_LINEAR pixel-centre mapping and edge replication.
        float sx = (float(gid.x - pad_left) + 0.5f) / scale - 0.5f;
        float sy = (float(gid.y - pad_top) + 0.5f) / scale - 0.5f;
        sx = clamp(sx, 0.0f, float(input_w - 1));
        sy = clamp(sy, 0.0f, float(input_h - 1));
        int x0 = int(floor(sx));
        int y0 = int(floor(sy));
        int x1 = min(x0 + 1, int(input_w - 1));
        int y1 = min(y0 + 1, int(input_h - 1));
        float fx = sx - float(x0);   // fractional part in x
        float fy = sy - float(y0);   // fractional part in y

        float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;

        // Unrolled 2×2 sampling loop — the Metal compiler will fully unroll
        // this because the loop bounds are compile-time constants.
        for (int dy = 0; dy <= 1; ++dy) {
            int py = (dy == 0) ? y0 : y1;
            float wy = (dy == 0) ? (1.0f - fy) : fy;   // row weight
            for (int dx = 0; dx <= 1; ++dx) {
                int px = (dx == 0) ? x0 : x1;
                float wx = (dx == 0) ? (1.0f - fx) : fx; // column weight
                float weight = wx * wy;
                uint idx = (uint(py) * input_w + uint(px)) * 3;
                // Input pixel layout: idx+0=B, idx+1=G, idx+2=R.
                b_acc += float(input[idx + 0]) * weight;
                g_acc += float(input[idx + 1]) * weight;
                r_acc += float(input[idx + 2]) * weight;
            }
        }

        // Normalise to [0,1] in-place.
        r = r_acc / 255.0f;
        g = g_acc / 255.0f;
        b = b_acc / 255.0f;
    }

    // Write channel-first layout: R plane → G plane → B plane.
    // ONNXRuntime expects NCHW: [1, 3, 640, 640].
    uint area = target * target;
    uint idx = gid.y * target + gid.x;
    output[idx]           = r;
    output[area + idx]    = g;
    output[2 * area + idx] = b;
}

/**
 * Bilinear letterbox preprocessing from a BGRA8Unorm Metal texture.
 *
 * Used for the zero-copy CVPixelBuffer path. The hardware texture sampler
 * provides free bilinear filtering and clamp-to-edge addressing, so the
 * kernel is much simpler than the buffer variant. BGRA input is swizzled
 * to RGB output (B→channel 2, G→channel 1, R→channel 0).
 */
kernel void letterbox_from_texture(
    texture2d<float, access::sample> input [[texture(0)]],
    device float* output  [[buffer(0)]],
    constant uint& target [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant uint& pad_left [[buffer(3)]],
    constant uint& pad_top  [[buffer(4)]],
    constant uint& src_w    [[buffer(5)]],
    constant uint& src_h    [[buffer(6)]],
    constant uint& scaled_w [[buffer(7)]],
    constant uint& scaled_h [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= target || gid.y >= target) return;

    float r, g, b;

    if (gid.x < pad_left || gid.x >= pad_left + scaled_w ||
        gid.y < pad_top || gid.y >= pad_top + scaled_h) {
        r = g = b = 114.0f / 255.0f;
    } else {
        float sx = (float(gid.x - pad_left) + 0.5f) / scale - 0.5f;
        float sy = (float(gid.y - pad_top) + 0.5f) / scale - 0.5f;
        sx = clamp(sx, 0.0f, float(src_w - 1));
        sy = clamp(sy, 0.0f, float(src_h - 1));
        constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
        float4 color = input.sample(s, float2(sx + 0.5f, sy + 0.5f));
        // BGRA8Unorm is exposed to shaders as logical RGBA components.
        r = color.r;
        g = color.g;
        b = color.b;
    }

    uint area = target * target;
    uint idx = gid.y * target + gid.x;
    output[idx]           = r;
    output[area + idx]    = g;
    output[2 * area + idx] = b;
}
)metal";

namespace safety_helmet {

// ---------------------------------------------------------------------------
// Pimpl — hides Objective-C / Metal types from the C++ header.
// Each field is an Objective-C object reference (id<Protocol>) managed by ARC.
// ---------------------------------------------------------------------------
struct MetalPreprocessor::Impl {
    id<MTLDevice>       device         = nil;  // MTLDevice (system default GPU)
    id<MTLCommandQueue> commandQueue   = nil;  // Serial command queue
    id<MTLComputePipelineState> pipelineBuffer = nil;  // Kernel 1 (buffer input)
    id<MTLComputePipelineState> pipelineTex    = nil;  // Kernel 2 (texture input)
    CVMetalTextureCacheRef textureCache = nullptr;    // CVPixelBuffer→MTLTexture bridge
    bool available = false;

    /// Releases all Metal objects. CVMetalTextureCache needs explicit CFRelease.
    ~Impl() {
        if (textureCache) {
            CFRelease(textureCache);
            textureCache = nullptr;
        }
        // ARC releases ObjC objects automatically.
        pipelineBuffer = nil;
        pipelineTex = nil;
        commandQueue = nil;
        device = nil;
    }
};

// ---------------------------------------------------------------------------
// Constructor — one-time Metal initialisation.
//
// Steps:
//   1. Acquire system default MTLDevice (M1/M2/M3 GPU or software renderer).
//   2. Create a serial command queue.
//   3. Compile embedded MSL source → MTLLibrary.
//   4. Extract both kernel functions → MTLComputePipelineState.
//   5. Create CVMetalTextureCache for zero-copy CVPixelBuffer access.
//
// Any failure sets `available = false`; the caller uses the CPU fallback.
// ---------------------------------------------------------------------------
MetalPreprocessor::MetalPreprocessor()
    : impl_(std::make_unique<Impl>())
{
    // Step 1: acquire GPU device.
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        ALGO_LOG_WARN("MetalPreprocessor: No Metal device found, falling back to CPU.");
        return;
    }

    // Step 2: serial command queue (one command buffer at a time per queue).
    // This is safe because the pipeline serialises frames via std::mutex.
    impl_->commandQueue = [impl_->device newCommandQueue];
    if (!impl_->commandQueue) {
        ALGO_LOG_WARN("MetalPreprocessor: Failed to create command queue.");
        impl_->device = nil;
        return;
    }

    // Step 3: compile MSL source at runtime.
    // We embed the shader as a C string to avoid needing Xcode/.metallib build steps.
    NSError* error = nil;
    NSString* src = [NSString stringWithUTF8String:kMetalShaderSource];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;

    id<MTLLibrary> library = [impl_->device newLibraryWithSource:src
                                                          options:opts
                                                            error:&error];
    if (!library) {
        ALGO_LOG_ERROR("MetalPreprocessor: Shader compilation failed: %s",
                       [[error localizedDescription] UTF8String]);
        impl_->device = nil;
        impl_->commandQueue = nil;
        return;
    }

    // Step 4: extract named functions and create compute pipeline states.
    id<MTLFunction> fnBuffer = [library newFunctionWithName:@"letterbox_from_buffer"];
    id<MTLFunction> fnTex    = [library newFunctionWithName:@"letterbox_from_texture"];

    if (!fnBuffer || !fnTex) {
        ALGO_LOG_ERROR("MetalPreprocessor: Kernel function not found in library.");
        impl_->device = nil;
        impl_->commandQueue = nil;
        return;
    }

    impl_->pipelineBuffer = [impl_->device newComputePipelineStateWithFunction:fnBuffer
                                                                         error:&error];
    if (!impl_->pipelineBuffer) {
        ALGO_LOG_ERROR("MetalPreprocessor: Pipeline creation (buffer) failed: %s",
                       [[error localizedDescription] UTF8String]);
        impl_->device = nil;
        impl_->commandQueue = nil;
        return;
    }

    impl_->pipelineTex = [impl_->device newComputePipelineStateWithFunction:fnTex
                                                                      error:&error];
    if (!impl_->pipelineTex) {
        ALGO_LOG_ERROR("MetalPreprocessor: Pipeline creation (texture) failed: %s",
                       [[error localizedDescription] UTF8String]);
        impl_->device = nil;
        impl_->commandQueue = nil;
        return;
    }

    // Step 5: create texture cache for zero-copy CVPixelBuffer bridging.
    // This cache maps CVPixelBuffer IOSurface → MTLTexture without copying.
    CVReturn cvRet = CVMetalTextureCacheCreate(
        kCFAllocatorDefault, nullptr, impl_->device, nullptr, &impl_->textureCache);
    if (cvRet != kCVReturnSuccess) {
        ALGO_LOG_WARN("MetalPreprocessor: CVMetalTextureCacheCreate failed (cv=%d), "
                      "CVPixelBuffer path disabled.", cvRet);
        impl_->textureCache = nullptr;
        // Not fatal — buffer path still works.
    }

    impl_->available = true;
    ALGO_LOG_INFO("MetalPreprocessor: GPU preprocessing initialized on %s.",
                  [[impl_->device name] UTF8String]);
}

MetalPreprocessor::~MetalPreprocessor() = default;

// ---------------------------------------------------------------------------
// Public query
// ---------------------------------------------------------------------------
bool MetalPreprocessor::IsAvailable() const {
    return impl_ && impl_->available;
}

// ---------------------------------------------------------------------------
// Process() — cv::Mat BGR → float tensor via Metal buffer path.
//
// Pipeline:
//   1. Compute letterbox geometry (same formula as CPU LetterBoxPreprocess).
//   2. Copy cv::Mat rows into a tightly packed shared MTLBuffer.
//      Ordinary OpenCV allocations do not satisfy newBufferWithBytesNoCopy's
//      VM allocation and page-alignment ownership contract.
//   3. Allocate output MTLBuffer (3 * target² * sizeof(float)).
//   4. Encode and dispatch compute kernel.
//   5. Commit, waitUntilCompleted, download output buffer.
//
// MTLResourceStorageModeShared ensures the buffer resides in Unified Memory
// and is accessible to both CPU and GPU without explicit synchronisation.
// ---------------------------------------------------------------------------
bool MetalPreprocessor::Process(const cv::Mat& bgr_image,
                                std::vector<float>& output_tensor,
                                int target_size,
                                LetterBoxInfo& box_info)
{
    if (!impl_ || !impl_->available) return false;
    if (bgr_image.empty() || bgr_image.type() != CV_8UC3) return false;

    int orig_w = bgr_image.cols;
    int orig_h = bgr_image.rows;

    // Compute letterbox parameters.
    // ratio = min(target/w, target/h) — fits the longer side into the square.
    float r = std::min(static_cast<float>(target_size) / orig_w,
                       static_cast<float>(target_size) / orig_h);
    float new_w = std::round(orig_w * r);
    float new_h = std::round(orig_h * r);
    float p_w = (target_size - new_w) / 2.0f;  // horizontal padding per side
    float p_h = (target_size - new_h) / 2.0f;  // vertical padding per side

    box_info.ratio = r;
    box_info.pad_x = p_w;
    box_info.pad_y = p_h;
    box_info.orig_w = orig_w;
    box_info.orig_h = orig_h;

    // Copy into a tightly packed shared buffer. newBufferWithBytesNoCopy is
    // invalid for ordinary cv::Mat allocations because Metal requires a
    // page-aligned VM allocation whose lifetime follows its deallocator.
    int row_bytes = orig_w * 3;
    size_t data_size = static_cast<size_t>(orig_h) * row_bytes;
    id<MTLBuffer> inputBuf = [impl_->device newBufferWithLength:data_size
                                                        options:MTLResourceStorageModeShared];
    if (!inputBuf) return false;
    uint8_t* dst = static_cast<uint8_t*>([inputBuf contents]);
    // Fast path: single memcpy when rows are tightly packed.
    // cv::Mat returned by cv::imread and freshly allocated images
    // are almost always continuous — this avoids N function-call
    // overhead of the row-by-row loop.
    if (bgr_image.isContinuous()) {
        std::memcpy(dst, bgr_image.data, data_size);
    } else {
        for (int y = 0; y < orig_h; ++y) {
            std::memcpy(dst + static_cast<size_t>(y) * row_bytes,
                        bgr_image.ptr(y), row_bytes);
        }
    }

    // Output buffer: 3 channels × target × target floats.
    size_t outSize = static_cast<size_t>(3 * target_size * target_size) * sizeof(float);
    id<MTLBuffer> outputBuf = [impl_->device newBufferWithLength:outSize
                                                         options:MTLResourceStorageModeShared];
    if (!outputBuf) return false;

    // Pack kernel parameters into stack variables (setBytes copies by value).
    uint32_t iw = static_cast<uint32_t>(orig_w);
    uint32_t ih = static_cast<uint32_t>(orig_h);
    uint32_t ts = static_cast<uint32_t>(target_size);
    uint32_t nw = static_cast<uint32_t>(new_w);
    uint32_t nh = static_cast<uint32_t>(new_h);
    uint32_t pl = static_cast<uint32_t>(std::round(p_w - 0.1f));
    uint32_t pt = static_cast<uint32_t>(std::round(p_h - 0.1f));
    float scl = r;

    id<MTLCommandBuffer> cmdBuf = [impl_->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:impl_->pipelineBuffer];
    [enc setBuffer:inputBuf  offset:0 atIndex:0];
    [enc setBuffer:outputBuf offset:0 atIndex:1];
    [enc setBytes:&iw  length:sizeof(uint32_t) atIndex:2];
    [enc setBytes:&ih  length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&ts  length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&scl length:sizeof(float)    atIndex:5];
    [enc setBytes:&pl  length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&pt  length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&nw  length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&nh  length:sizeof(uint32_t) atIndex:9];

    // Dispatch: one thread per output pixel.
    // 16×16 threadgroups give good occupancy on Apple GPU (up to 1024 threads/group).
    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(target_size, target_size, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];  // Synchronous — we need the result immediately.

    if (cmdBuf.status == MTLCommandBufferStatusError) {
        ALGO_LOG_ERROR("MetalPreprocessor: GPU command buffer error (buffer path).");
        return false;
    }

    // Download result from Unified Memory to host vector.
    output_tensor.resize(3 * target_size * target_size);
    std::memcpy(output_tensor.data(), [outputBuf contents], outSize);
    return true;
}

// ---------------------------------------------------------------------------
// ProcessFromPixelBuffer() — CVPixelBuffer → float tensor (zero-copy path).
//
// Currently only supports BGRA (kCVPixelFormatType_32BGRA) pixel buffers.
// For NV12 buffers, returns false so the caller falls back to CPU conversion.
//
// The CVPixelBufferRef is *borrowed* from the Engine; we wrap it in a
// temporary CVMetalTexture and release the texture (not the pixel buffer)
// before returning.
// ---------------------------------------------------------------------------
bool MetalPreprocessor::ProcessFromPixelBuffer(void* cvPixelBuffer,
                                                std::vector<float>& output_tensor,
                                                int target_size,
                                                LetterBoxInfo& box_info)
{
    if (!impl_ || !impl_->available || !impl_->textureCache) return false;
    if (!cvPixelBuffer) return false;

    CVPixelBufferRef pxBuf = static_cast<CVPixelBufferRef>(cvPixelBuffer);
    OSType pxFormat = CVPixelBufferGetPixelFormatType(pxBuf);
    size_t srcW = CVPixelBufferGetWidth(pxBuf);
    size_t srcH = CVPixelBufferGetHeight(pxBuf);

    // Only BGRA CVPixelBuffers are supported via texture cache currently.
    // NV12 would require YUV→RGB conversion in the shader (future work).
    if (pxFormat != kCVPixelFormatType_32BGRA) {
        ALGO_LOG_WARN("MetalPreprocessor: Unsupported CVPixelBuffer format 0x%x, "
                      "falling back to CPU.", pxFormat);
        return false;
    }

    // Compute letterbox parameters.
    float r = std::min(static_cast<float>(target_size) / static_cast<float>(srcW),
                       static_cast<float>(target_size) / static_cast<float>(srcH));
    float new_w = std::round(srcW * r);
    float new_h = std::round(srcH * r);
    float p_w = (target_size - new_w) / 2.0f;
    float p_h = (target_size - new_h) / 2.0f;

    box_info.ratio = r;
    box_info.pad_x = p_w;
    box_info.pad_y = p_h;
    box_info.orig_w = static_cast<int>(srcW);
    box_info.orig_h = static_cast<int>(srcH);

    // Zero-copy: map CVPixelBuffer's IOSurface backing to a Metal texture.
    // CVMetalTextureCache transparently handles format conversion and
    // synchronisation. The resulting texture aliases the pixel buffer memory.
    CVMetalTextureRef cvMetalTex = nullptr;
    CVReturn cvRet = CVMetalTextureCacheCreateTextureFromImage(
        kCFAllocatorDefault,
        impl_->textureCache,
        pxBuf,
        nullptr,
        MTLPixelFormatBGRA8Unorm,  // matches BGRA CVPixelBuffer
        srcW, srcH,
        0,   // plane index (0 for interleaved BGRA)
        &cvMetalTex);
    if (cvRet != kCVReturnSuccess || !cvMetalTex) {
        ALGO_LOG_ERROR("MetalPreprocessor: CVMetalTextureCacheCreateTextureFromImage "
                       "failed (cv=%d).", cvRet);
        return false;
    }

    id<MTLTexture> inputTex = CVMetalTextureGetTexture(cvMetalTex);
    if (!inputTex) {
        CFRelease(cvMetalTex);
        return false;
    }

    // Allocate output buffer.
    size_t outSize = static_cast<size_t>(3 * target_size * target_size) * sizeof(float);
    id<MTLBuffer> outputBuf = [impl_->device newBufferWithLength:outSize
                                                         options:MTLResourceStorageModeShared];
    if (!outputBuf) {
        CFRelease(cvMetalTex);
        return false;
    }

    // Kernel parameters.
    uint32_t ts = static_cast<uint32_t>(target_size);
    uint32_t sw = static_cast<uint32_t>(srcW);
    uint32_t sh = static_cast<uint32_t>(srcH);
    uint32_t nw = static_cast<uint32_t>(new_w);
    uint32_t nh = static_cast<uint32_t>(new_h);
    uint32_t pl = static_cast<uint32_t>(std::round(p_w - 0.1f));
    uint32_t pt = static_cast<uint32_t>(std::round(p_h - 0.1f));
    float scl = r;

    id<MTLCommandBuffer> cmdBuf = [impl_->commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:impl_->pipelineTex];
    [enc setTexture:inputTex  atIndex:0];
    [enc setBuffer:outputBuf  offset:0 atIndex:0];
    [enc setBytes:&ts  length:sizeof(uint32_t) atIndex:1];
    [enc setBytes:&scl length:sizeof(float)    atIndex:2];
    [enc setBytes:&pl  length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&pt  length:sizeof(uint32_t) atIndex:4];
    [enc setBytes:&sw  length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&sh  length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&nw  length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&nh  length:sizeof(uint32_t) atIndex:8];

    MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSize = MTLSizeMake(target_size, target_size, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [enc endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Release the temporary Metal texture view — the underlying CVPixelBuffer
    // remains valid because we never retained it.
    CFRelease(cvMetalTex);

    if (cmdBuf.status == MTLCommandBufferStatusError) {
        ALGO_LOG_ERROR("MetalPreprocessor: GPU command buffer error (pixel buffer path).");
        return false;
    }

    output_tensor.resize(3 * target_size * target_size);
    std::memcpy(output_tensor.data(), [outputBuf contents], outSize);
    return true;
}

} // namespace safety_helmet
