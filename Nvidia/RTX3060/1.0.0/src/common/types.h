/**
 * @file types.h
 * @brief GPU Face Recognition — Common data type definitions
 * @module Data Layer
 * @note  Core data structures for the GPU face recognition pipeline.
 *       All structures are POD-style for cross-module passing and serialization.
 *       Adapted from the Apple Silicon face_rec package for NVIDIA GPU.
 */

#ifndef GPU_FACE_RECOGNITION_TYPES_H
#define GPU_FACE_RECOGNITION_TYPES_H

#include <array>
#include <vector>
#include <string>
#include <cstdint>

namespace face_rec {

// ============================================================
// Category Code Constants
// ============================================================
/// @name Category Codes — values for category_code in inference results
///@{
constexpr int CAT_PERSON       = 13003;  // Person
constexpr int CAT_UNKNOWN_FACE = 13002;  // Unknown face
constexpr int CAT_KNOWN_FACE   = 13001;  // Known (identified) face
///@}

// ============================================================
// Pixel Format FourCC Codes
// ============================================================
/// @name Pixel Format FourCC — for hw_buffer_desc_t.pixel_format
///@{
namespace pixel_format {
    constexpr uint32_t BGR24 = ('B') | ('G' << 8) | ('R' << 16) | ('3' << 24);
    constexpr uint32_t NV12  = ('N') | ('V' << 8) | ('1' << 16) | ('2' << 24);
}
///@}

/**
 * @struct Point
 * @brief 2D coordinate point in image coordinate space.
 */
struct Point {
    float x = 0.0f;
    float y = 0.0f;
};

/**
 * @struct Rect
 * @brief Axis-aligned bounding rectangle.
 */
struct Rect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

/**
 * @struct Image
 * @brief Image data descriptor (non-owning view).
 *        Does not hold data ownership — wraps external pitch-linear data.
 */
struct Image {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int stride = 0;  // Bytes per row (may include padding)
};

/**
 * @struct DetectedObject
 * @brief Comprehensive data structure for a detected object (face or person).
 *        Flows through the entire pipeline: detection → tracking → alignment → extraction → recognition.
 */
struct DetectedObject {
    Rect bbox;            // Pixel coordinates (absolute)
    float confidence = 0.0f;
    int label = -1;
    std::array<Point, 5> landmarks{};  // 5 face landmarks (left eye, right eye, nose, left mouth, right mouth)
    int track_id = -1;
    std::vector<float> embedding;      // AdaFace 512-d normalized embedding

    // Face Recognition identity fields
    std::string identity_id;
    std::string identity_name;
    float similarity = 0.0f;

    struct Candidate {
        std::string identity_id;
        std::string identity_name;
        float similarity = 0.0f;
    };
    std::vector<Candidate> candidates;

    Rect person_bbox;  // Associated body bounding box
};

/**
 * @struct CropParams
 * @brief Image crop region parameters (integer coordinates).
 */
struct CropParams {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_TYPES_H
