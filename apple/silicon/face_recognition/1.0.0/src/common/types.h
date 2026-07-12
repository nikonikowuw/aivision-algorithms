/**
 * @file types.h
 * @brief 人脸识别算法包通用数据类型定义
 *        Common data type definitions for the face recognition algorithm package.
 * @module 基础数据层 (Data Layer)
 * @note 本文件定义了算法流水线中各阶段使用的核心数据结构，
 *       包括坐标点、矩形框、图像描述、检测结果及裁剪参数。
 *       所有结构均采用 POD 风格以方便跨模块传递和序列化。
 */

#ifndef FACE_RECOGNITION_TYPES_H
#define FACE_RECOGNITION_TYPES_H

#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace face_rec {

// ============================================================
// 分类编码常量 (Category Code Constants)
// ============================================================
/// @name Category Codes — 推理结果中的 category_code 字段取值
///@{
constexpr int CAT_PERSON       = 13003;  // 人体 / Person
constexpr int CAT_UNKNOWN_FACE = 13002;  // 未知人脸 / Unknown face
constexpr int CAT_KNOWN_FACE   = 13001;  // 已知人脸 / Known (identified) face
///@}

// ============================================================
// 像素格式 FourCC (Pixel Format FourCC Codes)
// ============================================================
/// @name Pixel Format FourCC — 用于 hw_buffer_desc_t.pixel_format
///@{
namespace pixel_format {
    constexpr uint32_t BGR24 = ('B') | ('G' << 8) | ('R' << 16) | ('3' << 24);
    constexpr uint32_t NV12  = ('N') | ('V' << 8) | ('1' << 16) | ('2' << 24);
}
///@}

/**
 * @struct Point
 * @brief 二维坐标点 (2D point in image coordinate space)
 *        用于表示面部关键点、边界框角点等位置信息。
 *        坐标原点为图像左上角，x 轴向右，y 轴向下。
 */
struct Point {
    float x = 0.0f;  // X coordinate (column)
    float y = 0.0f;  // Y coordinate (row)
};

/**
 * @struct Rect
 * @brief 矩形边界框 (Axis-aligned bounding rectangle)
 *        以 (x, y) 为左上角坐标，width/height 为宽高。
 *        所有值均为浮点型，可表示子像素精度。
 */
struct Rect {
    float x = 0.0f;      // Top-left X
    float y = 0.0f;      // Top-left Y
    float width = 0.0f;  // Width (pixels)
    float height = 0.0f; // Height (pixels)
};

/**
 * @struct Image
 * @brief 图像数据描述符 (Image descriptor, non-owning view)
 *        不持有数据所有权，仅指向外部图像缓冲区。
 *        适用于从 Metal/ANE 等零拷贝流程传入的图像数据。
 * @note stride 表示一行数据的实际字节跨度（可能大于 width*channels 的对齐值）
 */
struct Image {
    const uint8_t* data = nullptr;  // Pointer to raw pixel data
    int width = 0;                  // Pixel width
    int height = 0;                 // Pixel height
    int channels = 0;               // Number of color channels (e.g. 3 for RGB, 4 for RGBA)
    int stride = 0;                 // Bytes per row (row stride, may include padding)
};

/**
 * @struct DetectedObject
 * @brief 检测到的目标对象（人脸或人体）综合数据结构
 *        Comprehensive data structure for a detected object (face or person).
 * @details 贯穿整个算法流水线：检测 → 跟踪 → 对齐 → 特征提取 → 识别。
 *          包含检测框、关键点、跟踪 ID、特征向量及识别结果等字段。
 */
struct DetectedObject {
    Rect bbox;            // Pixel coordinates (absolute)
    float confidence = 0.0f;              // Detection confidence score (0.0 ~ 1.0)
    int label = -1;                       // Class label (e.g. 0=person, 1=face)
    std::array<Point, 5> landmarks{};     // 5 face landmarks (left eye, right eye, nose, left mouth, right mouth)
    int track_id = -1;                    // Track ID associated by the tracker
    std::vector<float> embedding;         // AdaFace 512-d normalized embedding

    // Face Recognition identity fields
    std::string identity_id;              // Matched identity ID from the gallery
    std::string identity_name;            // Matched identity display name
    float similarity = 0.0f;              // Cosine similarity score with the matched identity

    /**
     * @struct Candidate
     * @brief 候选识别结果（Top-K 匹配）
     *        用于支持多候选返回，排序依据 similarity 降序。
     */
    struct Candidate {
        std::string identity_id;          // Candidate identity ID
        std::string identity_name;        // Candidate display name
        float similarity = 0.0f;          // Similarity score with this candidate
    };
    std::vector<Candidate> candidates;    // Top-K ranked candidate list

    Rect person_bbox;                     // Associated body bounding box
};

/**
 * @struct CropParams
 * @brief 图像裁剪参数 (Image crop region parameters)
 *        定义从源图像中裁剪一个矩形区域的坐标和尺寸。
 *        与 Rect 的区别在于使用整数坐标，直接用于像素级裁剪操作。
 */
struct CropParams {
    int x = 0;  // Crop origin X (column offset)
    int y = 0;  // Crop origin Y (row offset)
    int w = 0;  // Crop width
    int h = 0;  // Crop height
};

} // namespace face_rec

#endif // FACE_RECOGNITION_TYPES_H
