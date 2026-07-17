/**
 * @file types.h
 * @brief People Counting — Common data type definitions
 * @module Data Layer
 */

#ifndef PEOPLE_COUNTING_TYPES_H
#define PEOPLE_COUNTING_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace people_count {

// ============================================================
// Category Code Constants
// ============================================================
constexpr int CAT_PERSON = 10001;  // Person category code

// ============================================================
// Pixel Format FourCC Codes
// ============================================================
namespace pixel_format {
    constexpr uint32_t BGR24 = ('B') | ('G' << 8) | ('R' << 16) | ('3' << 24);
    constexpr uint32_t NV12  = ('N') | ('V' << 8) | ('1' << 16) | ('2' << 24);
}

/**
 * @struct Point
 * @brief 2D coordinate point
 */
struct Point {
    float x = 0.0f;
    float y = 0.0f;
};

/**
 * @struct Rect
 * @brief Bounding box
 */
struct Rect {
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

/**
 * @struct Image
 * @brief Non-owning view of image data
 */
struct Image {
    const uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
    int stride = 0;
};

/**
 * @struct DetectedObject
 * @brief A detected and tracked person object
 */
struct DetectedObject {
    Rect bbox;            // Pixel coordinates (absolute or normalized depending on stage)
    float confidence = 0.0f;
    int label = -1;
    int track_id = -1;
    bool crossed = false;
    std::string direction = "none";
};

} // namespace people_count

#endif // PEOPLE_COUNTING_TYPES_H
