/**
 * @file test_e2e_infer.cpp
 * @brief 端到端推理测试 — 使用合成 BGR24 图像验证完整推理管线
 *        End-to-End Inference Test — validates the full inference pipeline with a synthetic BGR24 frame
 *
 * 测试场景：
 *   1. 合成 640×480 BGR24 灰色图像 / Generate a 640×480 synthetic gray BGR24 image
 *   2. 加载动态库，dlsym 获取所有 ABI 符号 / Load .so, fetch ABI symbols via dlsym
 *   3. detector_init 初始化算法上下文 / Initialize algorithm context
 *   4. detector_infer 执行推理 / Run inference
 *   5. 解析并打印 JSON 推理结果 / Parse and print JSON inference results
 *   6. algo_free_result + detector_destroy 清理 / Cleanup
 */

#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>
#import <CoreText/CoreText.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <string>
#include <vector>
#include "algo/abi_contract.h"

// 像素格式 FourCC (应与 src/common/types.h 中 pixel_format::* 一致)
// Pixel format FourCC (should match pixel_format::* in src/common/types.h)
constexpr uint32_t PIX_FMT_BGR24 = ('B') | ('G' << 8) | ('R' << 16) | ('3' << 24);

constexpr int FRAME_W = 640;
constexpr int FRAME_H = 480;
constexpr int BYTES_PER_PIXEL = 3;
constexpr int STRIDE = FRAME_W * BYTES_PER_PIXEL;  // 1920
constexpr int FRAME_SIZE = FRAME_H * STRIDE;        // 921600

// 函数指针类型 (Function pointer types matching the C ABI)
typedef algo_handle_t (*detector_init_func)(const char* config_json);
typedef int (*detector_infer_func)(algo_handle_t handle, const hw_buffer_desc_t* input,
                                   const char* context_json, infer_result_t* result);
typedef int (*detector_update_face_library_func)(algo_handle_t handle, const char* face_library_json);
typedef void (*algo_free_result_func)(infer_result_t* result);
typedef void (*detector_destroy_func)(algo_handle_t handle);
typedef int (*detector_self_test_func)(void);
typedef const char* (*detector_version_func)(void);
typedef const char* (*detector_name_func)(void);

/**
 * @brief 找到与位置 pos 处的 { 匹配的 }，支持嵌套和字符串转义
 *        Find the matching closing brace for { at pos (handles nesting and string escapes)
 */
static size_t FindMatchingBrace(const std::string& json, size_t pos) {
    if (pos >= json.size() || json[pos] != '{') return std::string::npos;
    int depth = 1;
    bool in_string = false;
    for (size_t i = pos + 1; i < json.size(); i++) {
        char c = json[i];
        if (in_string && c == '\\') { i++; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (!in_string) {
            if (c == '{') { depth++; }
            else if (c == '}') { depth--; if (depth == 0) return i; }
        }
    }
    return std::string::npos;
}

/**
 * @brief 将 JSON 检测结果绘制到图像上并保存为 result.jpg
 *        Draw detection results on the image and save as result.jpg
 *
 * @param frame_buf  原始 BGR24 图像数据 / Raw BGR24 image data
 * @param frame_w    图像宽度 / Image width
 * @param frame_h    图像高度 / Image height
 * @param frame_stride 图像行步长 / Image row stride
 * @param json       JSON 检测结果字符串 / Detection result JSON string
 */
static void DrawAndSaveResult(const std::vector<uint8_t>& frame_buf,
                               int frame_w, int frame_h, int frame_stride,
                               const std::string& json) {
    // 转换为 BGRA（CGContext 需要 4 字节/像素）
    // Convert BGR24 to BGRA (CGContext requires 4 bytes/pixel)
    std::vector<uint8_t> bgra_buf(frame_h * frame_w * 4);
    for (int y = 0; y < frame_h; y++) {
        for (int x = 0; x < frame_w; x++) {
            int src_idx = y * frame_stride + x * 3;
            int dst_idx = y * frame_w * 4 + x * 4;
            bgra_buf[dst_idx]     = frame_buf[src_idx];     // B
            bgra_buf[dst_idx + 1] = frame_buf[src_idx + 1]; // G
            bgra_buf[dst_idx + 2] = frame_buf[src_idx + 2]; // R
            bgra_buf[dst_idx + 3] = 255;                     // A
        }
    }

    // 创建 CGContext（BGRA little-endian）
    // Create CGContext (BGRA little-endian)
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate(
        bgra_buf.data(), frame_w, frame_h, 8, frame_w * 4,
        colorSpace,
        kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little
    );
    CGColorSpaceRelease(colorSpace);
    if (!ctx) {
        printf("  ✗ Failed to create CGContext for result.jpg\n");
        return;
    }

    // 设置绘图参数 / Set drawing parameters
    CGContextSetLineWidth(ctx, 3.0f);

    // 创建 CoreText 字体和白色 / CoreText font and white color
    CFStringRef fontName = CFSTR("Helvetica");
    CTFontRef ctFont = CTFontCreateWithName(fontName, 14.0, NULL);
    if (!ctFont) {
        CGContextRelease(ctx);
        printf("  ✗ Failed to create CTFont for result.jpg\n");
        return;
    }
    CGColorSpaceRef rgbCS = CGColorSpaceCreateDeviceRGB();
    CGFloat whiteComps[] = {1.0, 1.0, 1.0, 1.0};
    CGColorRef whiteCG = CGColorCreate(rgbCS, whiteComps);
    CGColorSpaceRelease(rgbCS);

    // 解析 JSON 并逐对象绘制 / Parse JSON and draw each object
    size_t pos = 0;
    int obj_count = 0;
    while ((pos = json.find("{", pos)) != std::string::npos) {
        size_t end = FindMatchingBrace(json, pos);
        if (end == std::string::npos) break;
        std::string obj = json.substr(pos, end - pos + 1);

        // 提取字段 / Extract fields
        std::string label;
        float conf = 0, bx = 0, by = 0, bw = 0, bh = 0;

        size_t lp = obj.find("\"label\"");
        if (lp != std::string::npos) {
            size_t q1 = obj.find('"', lp + 8);
            size_t q2 = obj.find('"', q1 + 1);
            if (q1 != std::string::npos && q2 != std::string::npos)
                label = obj.substr(q1 + 1, q2 - q1 - 1);
        }

        // 跳过没有 label 的嵌套对象（如 landmarks、person_bbox 等）
        // Skip nested objects without label (landmarks, person_bbox, etc.)
        if (label.empty()) {
            pos = end + 1;
            continue;
        }

        // 解析 conf / Parse confidence
        size_t cp = obj.find("\"detect_confidence\"");
        if (cp != std::string::npos) {
            size_t col = obj.find(':', cp);
            size_t ns = obj.find_first_not_of(" \t", col + 1);
            size_t ne = obj.find_first_of(",}", ns);
            if (ns != std::string::npos && ne != std::string::npos)
                conf = std::stof(obj.substr(ns, ne - ns));
        }

        // 解析 bbox / Parse bbox
        size_t bp = obj.find("\"bbox\"");
        if (bp != std::string::npos) {
            // 找 x, y, w, h
            auto find_field = [&](const std::string& field) -> float {
                size_t fp = obj.find("\"" + field + "\"", bp);
                if (fp == std::string::npos) return 0.0f;
                size_t col = obj.find(':', fp);
                size_t ns = obj.find_first_not_of(" \t", col + 1);
                size_t ne = obj.find_first_of(",}", ns);
                if (ns == std::string::npos || ne == std::string::npos) return 0.0f;
                return std::stof(obj.substr(ns, ne - ns));
            };
            bx = find_field("x");
            by = find_field("y");
            bw = find_field("w");
            bh = find_field("h");
        }

        // 归一化坐标转像素（JSON 为左上角原点，而 CoreGraphics 默认使用左下角原点，Y 轴需做转换）
        // Convert normalized coords to pixels (JSON uses top-left origin, CoreGraphics uses bottom-left origin)
        float px = bx * frame_w;
        float py = by * frame_h;
        float pw = bw * frame_w;
        float ph = bh * frame_h;

        // CoreGraphics 原点在左下角，矩形底边 Y 坐标为 frame_h - py - ph
        // CoreGraphics origin is bottom-left, rect bottom Y is frame_h - py - ph
        float cg_y = frame_h - py - ph;

        // 根据标签选颜色 / Choose color by label
        CGFloat r = 0.0f, g = 1.0f, b = 0.0f;  // 默认绿色 / default green (person)
        if (label.find("face") != std::string::npos) {
            r = 0.0f; g = 0.6f; b = 1.0f;  // 浅蓝 / light blue (face)
            if (label.find("known") != std::string::npos) {
                r = 1.0f; g = 0.2f; b = 0.2f;  // 红色 / red (known face)
            }
        }

        // 绘制矩形框 / Draw bounding box
        CGContextSetRGBStrokeColor(ctx, r, g, b, 1.0f);
        CGContextStrokeRect(ctx, CGRectMake(px, cg_y, pw, ph));

        // 在顶部绘制标签背景条 / Draw label background bar
        std::string label_text = label + " " + std::to_string((int)(conf * 100)) + "%";
        CGFloat bar_w = pw;
        if (bar_w < 10) bar_w = label_text.size() * 10;  // 小框扩展标签
        CGContextSetRGBFillColor(ctx, r, g, b, 0.8f);
        CGContextFillRect(ctx, CGRectMake(px, frame_h - py, bar_w, 18));

        // 绘制白色文字标签 / Draw white label text
        CGContextSetRGBFillColor(ctx, 1.0f, 1.0f, 1.0f, 1.0f);

        // 使用 CoreText 绘制标签 / Draw label using CoreText
        CFStringRef cfLabelText = CFStringCreateWithCString(NULL, label_text.c_str(), kCFStringEncodingUTF8);
        if (cfLabelText) {
            CFMutableAttributedStringRef attrStr = CFAttributedStringCreateMutable(NULL, 0);
            if (attrStr) {
                CFAttributedStringReplaceString(attrStr, CFRangeMake(0, 0), cfLabelText);
                CFAttributedStringSetAttribute(attrStr, CFRangeMake(0, CFStringGetLength(cfLabelText)),
                    kCTFontAttributeName, ctFont);
                CFAttributedStringSetAttribute(attrStr, CFRangeMake(0, CFStringGetLength(cfLabelText)),
                    kCTForegroundColorAttributeName, whiteCG);

                CTLineRef line = CTLineCreateWithAttributedString(attrStr);
                if (line) {
                    CGContextSetTextPosition(ctx, px + 2, frame_h - py + 4);
                    CTLineDraw(line, ctx);
                    CFRelease(line);
                }
                CFRelease(attrStr);
            }
            CFRelease(cfLabelText);
        }

        // 如果是 face 且有 person_bbox，用绿色实线画人体框
        // If face has person_bbox, draw person box in solid green
        if (label.find("face") != std::string::npos) {
            size_t pp = obj.find("\"person_bbox\"");
            if (pp != std::string::npos) {
                auto find_field = [&](const std::string& field) -> float {
                    size_t fp = obj.find("\"" + field + "\"", pp);
                    if (fp == std::string::npos) return 0.0f;
                    size_t col = obj.find(':', fp);
                    size_t ns = obj.find_first_not_of(" \t", col + 1);
                    size_t ne = obj.find_first_of(",}", ns);
                    if (ns == std::string::npos || ne == std::string::npos) return 0.0f;
                    return std::stof(obj.substr(ns, ne - ns));
                };
                float pbx = find_field("x") * frame_w;
                float pby = find_field("y") * frame_h;
                float pbw = find_field("w") * frame_w;
                float pbh = find_field("h") * frame_h;
                // 绿色实线画人体框 / Green solid line for person body
                CGContextSetRGBStrokeColor(ctx, 0.0f, 1.0f, 0.0f, 1.0f);
                CGContextStrokeRect(ctx, CGRectMake(pbx, frame_h - pby - pbh, pbw, pbh));
            }
        }

        obj_count++;
        pos = end + 1;
    }

    if (ctFont) CFRelease(ctFont);
    if (whiteCG) CFRelease(whiteCG);

    // 从 CGContext 创建 CGImage → 写入 JPEG
    // Create CGImage from CGContext → write JPEG
    CGImageRef cgImage = CGBitmapContextCreateImage(ctx);
    if (cgImage) {
        NSURL *outUrl = [NSURL fileURLWithPath:@"result.jpg"];
        CGImageDestinationRef dest = CGImageDestinationCreateWithURL(
            (CFURLRef)outUrl, CFSTR("public.jpeg"), 1, NULL);
        if (dest) {
            NSDictionary *props = @{
                (NSString*)kCGImageDestinationLossyCompressionQuality: @0.95f
            };
            CGImageDestinationAddImage(dest, cgImage, (CFDictionaryRef)props);
            CGImageDestinationFinalize(dest);
            CFRelease(dest);
            printf("  ✓ Saved result.jpg (%d objects drawn)\n", obj_count);
        }
        CGImageRelease(cgImage);
    }
    CGContextRelease(ctx);
}

/* ---- 已有的 JSON 解析和主函数 ---- */

/**
 * 解析 JSON 字符串中的整数值 (轻量级，无外部依赖)
 * Parse an integer field from a JSON string (lightweight, no external dependency)
 */
static int ParseJsonInt(const std::string& json, const std::string& key, size_t start, size_t* out_end) {
    std::string search = "\"" + key + "\"";
    size_t key_pos = json.find(search, start);
    if (key_pos == std::string::npos) return 0;

    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return 0;

    size_t num_start = json.find_first_not_of(" \t\r\n", colon + 1);
    if (num_start == std::string::npos) return 0;

    size_t num_end = json.find_first_of(",}] \t\r\n", num_start);
    std::string num_str = json.substr(num_start, num_end - num_start);
    if (out_end) *out_end = num_end;
    return std::stoi(num_str);
}

/**
 * 解析 JSON 字符串中的浮点数值
 * Parse a float field from a JSON string
 */
static float ParseJsonFloat(const std::string& json, const std::string& key, size_t start, size_t* out_end) {
    std::string search = "\"" + key + "\"";
    size_t key_pos = json.find(search, start);
    if (key_pos == std::string::npos) return 0.0f;

    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return 0.0f;

    size_t num_start = json.find_first_not_of(" \t\r\n", colon + 1);
    if (num_start == std::string::npos) return 0.0f;

    size_t num_end = json.find_first_of(",}] \t\r\n", num_start);
    std::string num_str = json.substr(num_start, num_end - num_start);
    if (out_end) *out_end = num_end;
    return std::stof(num_str);
}

/**
 * 解析 JSON 字符串中的字符串值 (提取引号内内容)
 * Parse a string field from a JSON string (extract content between quotes)
 */
static std::string ParseJsonString(const std::string& json, const std::string& key, size_t start, size_t* out_end) {
    std::string search = "\"" + key + "\"";
    size_t key_pos = json.find(search, start);
    if (key_pos == std::string::npos) return "";

    size_t colon = json.find(':', key_pos + search.size());
    if (colon == std::string::npos) return "";

    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";

    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";

    if (out_end) *out_end = q2 + 1;
    return json.substr(q1 + 1, q2 - q1 - 1);
}

/**
 * 打印推理结果 JSON (人类可读格式)
 * Print inference result JSON in human-readable format
 */
static void PrintResults(const infer_result_t& result) {
    if (!result.result_json || result.result_json_len == 0) {
        printf("  (no result data)\n");
        return;
    }

    std::string json(result.result_json, result.result_json_len);
    printf("  Inference time: %u µs\n", result.infer_time_us);
    printf("  JSON length: %zu\n", result.result_json_len);

    // 统计对象个数 / Count objects
    int object_count = 0;
    size_t pos = 0;
    while ((pos = json.find("\"category_code\"", pos)) != std::string::npos) {
        object_count++;
        pos += 15;
    }

    printf("  Detected objects: %d\n", object_count);

    // 输出原始 JSON 结果（即使空数组也展示）
    // Print raw JSON result (even empty array)
    printf("  Raw JSON: %s\n", json.c_str());
    printf("\n");

    if (object_count == 0) {
        return;
    }

    // 逐对象解析 / Parse each object
    size_t obj_start = 0;
    for (int i = 0; i < object_count; i++) {
        size_t brace_start = json.find('{', obj_start);
        size_t brace_end = FindMatchingBrace(json, brace_start);
        if (brace_start == std::string::npos || brace_end == std::string::npos) break;

        std::string obj = json.substr(brace_start, brace_end - brace_start + 1);

        std::string label = ParseJsonString(obj, "label", 0, nullptr);
        int cat = ParseJsonInt(obj, "category_code", 0, nullptr);
        float conf = ParseJsonFloat(obj, "detect_confidence", 0, nullptr);
        int tid = ParseJsonInt(obj, "track_id", 0, nullptr);

        printf("  [%d] %-16s cat=%5d conf=%.3f track=%3d",
               i, label.c_str(), cat, conf, tid);

        // 尝试解析 bbox / Parse bbox
        size_t bbox_pos = obj.find("\"bbox\"");
        if (bbox_pos != std::string::npos) {
            float bx = ParseJsonFloat(obj, "x", bbox_pos, nullptr);
            float by = ParseJsonFloat(obj, "y", bbox_pos, nullptr);
            float bw = ParseJsonFloat(obj, "w", bbox_pos, nullptr);
            float bh = ParseJsonFloat(obj, "h", bbox_pos, nullptr);
            printf(" bbox=(%.3f,%.3f,%.3f,%.3f)", bx, by, bw, bh);
        }
        printf("\n");

        // 人脸额外字段 / Face-specific fields
        if (cat == 13001 || cat == 13002) {
            size_t lm_pos = obj.find("\"landmarks\"");
            int lm_count = 0;
            if (lm_pos != std::string::npos) {
                size_t arr_start = obj.find('[', lm_pos);
                size_t arr_end = obj.find(']', arr_start + 1);
                if (arr_start != std::string::npos && arr_end != std::string::npos) {
                    std::string arr = obj.substr(arr_start, arr_end - arr_start);
                    for (size_t p = 0; (p = arr.find("\"x\"", p)) != std::string::npos; p += 10) {
                        lm_count++;
                    }
                }
            }

            float face_q = ParseJsonFloat(obj, "face_quality_score", 0, nullptr);
            printf("        face_quality=%.3f landmarks=%d\n", face_q, lm_count);

            if (cat == 13001) {  // known_face
                std::string id = ParseJsonString(obj, "identity_name", 0, nullptr);
                float sim = ParseJsonFloat(obj, "similarity", 0, nullptr);
                printf("        identity=\"%s\" similarity=%.3f\n", id.c_str(), sim);
            }
        }

        obj_start = brace_end + 1;
    }
}

/**
 * 从 raw BGR 文件中加载图像（文件格式：width(uint32_t) + height(uint32_t) + BGR24 pixels）
 * Load image from raw BGR file (format: width(uint32_t) + height(uint32_t) + BGR24 data)
 */
static bool LoadRawBgrFromFile(const char* path, std::vector<uint8_t>& buf,
                                int& width, int& height, int& stride) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return false; }

    uint32_t w = 0, h = 0;
    if (fread(&w, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1) {
        fclose(f); return false;
    }
    width = static_cast<int>(w);
    height = static_cast<int>(h);
    stride = width * 3;
    size_t expected = static_cast<size_t>(height) * stride;
    buf.resize(expected);
    size_t read = fread(buf.data(), 1, expected, f);
    fclose(f);
    return read == expected;
}

/**
 * 合成 BGR24 灰色图像缓冲
 * Generate a synthetic BGR24 gray image buffer
 */
static void FillSyntheticBgr(uint8_t* buf, int width, int height, int stride) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * stride + x * 3;
            buf[idx]     = 128;  // B
            buf[idx + 1] = 128;  // G
            buf[idx + 2] = 128;  // R
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <nikoniko_detector.so> [--raw-bgr <path>] [package_dir]\n", argv[0]);
        return 1;
    }

    const char* so_path = argv[1];
    const char* raw_bgr_path = nullptr;
    std::string pkg_dir;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--raw-bgr") == 0 && i + 1 < argc) {
            raw_bgr_path = argv[++i];
            continue;
        }
        if (pkg_dir.empty()) {
            pkg_dir = argv[i];
        }
    }

    printf("============================================================\n");
    printf("  Face Recognition — End-to-End Inference Test\n");
    printf("============================================================\n");
    printf("  Library: %s\n", so_path);
    printf("  Package: %s\n", pkg_dir.empty() ? "(auto-detect)" : pkg_dir.c_str());
    printf("\n");

    // ── 1. dlopen 加载动态库 / Load shared library via dlopen ──────
    printf("--- Step 1: Load Shared Library ---\n");
    void* so = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        fprintf(stderr, "✗ dlopen failed: %s\n", dlerror());
        return 1;
    }
    printf("  ✓ dlopen succeeded\n");
    printf("\n");

    // ── 2. dlsym 获取所有 ABI 符号 / Fetch all ABI symbols ─────────
    printf("--- Step 2: Fetch ABI Symbols ---\n");
    auto init_fn = reinterpret_cast<detector_init_func>(dlsym(so, "detector_init"));
    auto infer_fn = reinterpret_cast<detector_infer_func>(dlsym(so, "detector_infer"));
    auto update_face_library_fn = reinterpret_cast<detector_update_face_library_func>(dlsym(so, "detector_update_face_library"));
    auto free_fn = reinterpret_cast<algo_free_result_func>(dlsym(so, "algo_free_result"));
    auto destroy_fn = reinterpret_cast<detector_destroy_func>(dlsym(so, "detector_destroy"));
    auto self_test_fn = reinterpret_cast<detector_self_test_func>(dlsym(so, "detector_self_test"));
    auto version_fn = reinterpret_cast<detector_version_func>(dlsym(so, "detector_version"));
    auto name_fn = reinterpret_cast<detector_name_func>(dlsym(so, "detector_name"));

    if (!init_fn || !infer_fn || !update_face_library_fn || !free_fn || !destroy_fn || !self_test_fn || !version_fn || !name_fn) {
        fprintf(stderr, "✗ Missing ABI symbols\n");
        dlclose(so);
        return 1;
    }
    printf("  ✓ All 8 ABI symbols loaded\n");
    printf("  name=%s\n", name_fn());
    printf("  version=%s\n", version_fn());
    printf("\n");

    // ── 3. 自检 / Self-test ──────────────────────────────────────
    printf("--- Step 3: Self-Test ---\n");
    int ret = self_test_fn();
    if (ret != 0) {
        fprintf(stderr, "✗ detector_self_test failed: ret=%d\n", ret);
        dlclose(so);
        return 1;
    }
    printf("  ✓ detector_self_test passed\n");
    printf("\n");

    // ── 4. 初始化 / Initialize ──────────────────────────────────
    printf("--- Step 4: Initialize Algorithm Context ---\n");
    std::string config_str;
    if (!pkg_dir.empty()) {
        config_str = "{\"package_dir\":\"" + pkg_dir + "\"}";
    } else {
        config_str = "{}";
    }

    algo_handle_t handle = init_fn(config_str.c_str());
    if (!handle) {
        fprintf(stderr, "✗ detector_init returned NULL\n");
        dlclose(so);
        return 1;
    }
    printf("  ✓ handle = %p\n", static_cast<void*>(handle));
    printf("\n");

    // ── 5. 准备 BGR24 帧 / Prepare BGR24 frame ─────────────────────
    std::vector<uint8_t> frame_buf;
    int frame_w, frame_h, frame_stride;

    if (raw_bgr_path) {
        printf("--- Step 5: Load BGR24 Frame from File ---\n");
        printf("  Source: %s\n", raw_bgr_path);
        if (!LoadRawBgrFromFile(raw_bgr_path, frame_buf, frame_w, frame_h, frame_stride)) {
            fprintf(stderr, "✗ Failed to load raw BGR file: %s\n", raw_bgr_path);
            destroy_fn(handle);
            dlclose(so);
            return 1;
        }
    } else {
        printf("--- Step 5: Prepare Synthetic BGR24 Frame ---\n");
        frame_w = FRAME_W;
        frame_h = FRAME_H;
        frame_stride = STRIDE;
        frame_buf.resize(FRAME_SIZE, 128);
        FillSyntheticBgr(frame_buf.data(), frame_w, frame_h, frame_stride);
    }

    hw_buffer_desc_t hw_desc{};
    hw_desc.dma_fd = -1;
    hw_desc.size = static_cast<size_t>(frame_h) * frame_stride;
    hw_desc.width = static_cast<uint32_t>(frame_w);
    hw_desc.height = static_cast<uint32_t>(frame_h);
    hw_desc.pixel_format = PIX_FMT_BGR24;
    hw_desc.dma_buf_fd = -1;
    hw_desc.phys_addr = 0;
    hw_desc.data = frame_buf.data();
    hw_desc.stride = static_cast<uint32_t>(frame_stride);
    hw_desc.buffer_type = 0;
    hw_desc.buffer_owner = 0;
    hw_desc.reserved_flags = 0;
    // 清零联合体和填充 (zero out union & padding)
    std::memset(&hw_desc.plat, 0, sizeof(hw_desc.plat));

    printf("  Resolution: %d×%d\n", frame_w, frame_h);
    printf("  Format: BGR24 (FourCC 0x%08X)\n", PIX_FMT_BGR24);
    printf("  Buffer: %zu bytes, stride=%d\n", hw_desc.size, frame_stride);
    printf("  sizeof(hw_buffer_desc_t) = %zu (expected 144 on M1)\n", sizeof(hw_buffer_desc_t));
    printf("  sizeof(infer_result_t)   = %zu (expected 40 on M1)\n", sizeof(infer_result_t));
    printf("\n");

    // ── 6. 执行推理 / Run Inference ──────────────────────────────
    printf("--- Step 6: detector_infer ---\n");
    infer_result_t result{};
    result.result_json = nullptr;
    result.result_json_len = 0;
    result.infer_time_us = 0;

    ret = infer_fn(handle, &hw_desc, nullptr, &result);
    printf("  return code = %d\n", ret);

    if (ret != 0) {
        fprintf(stderr, "✗ Inference failed with error code %d\n", ret);
        destroy_fn(handle);
        dlclose(so);
        return 1;
    }
    printf("\n");

    // ── 7. 解析并打印结果 / Parse and Print Results ───────────
    printf("--- Step 7: Output ---\n");
    PrintResults(result);

    // 绘制并保存结果图像 / Draw and save result image
    if (result.result_json && result.result_json_len > 0) {
        std::string json(result.result_json, result.result_json_len);
        DrawAndSaveResult(frame_buf, frame_w, frame_h, frame_stride, json);
    }
    printf("\n");

    // ── 8. 清理 / Cleanup ─────────────────────────────────────
    printf("--- Step 8: Cleanup ---\n");
    free_fn(&result);
    destroy_fn(handle);
    dlclose(so);
    printf("  ✓ Cleanup complete\n");
    printf("\n");

    printf("============================================================\n");
    printf("  ✅ End-to-End Inference Test Passed\n");
    printf("============================================================\n");
    return 0;
}
