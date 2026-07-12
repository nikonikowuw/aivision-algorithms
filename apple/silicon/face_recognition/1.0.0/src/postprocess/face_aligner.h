/**
 * @file face_aligner.h
 * @brief 人脸对齐器接口与 InsightFace 实现声明
 *        Declarations of face aligner interface and InsightFace implementation
 *
 * 定义人脸对齐抽象接口，并提供基于 InsightFace 参考点的人脸对齐实现。
 * 对齐过程通过相似变换将检测到的人脸关键点映射到标准位置，
 * 生成 112×112 的归一化人脸图像，用于后续特征提取。
 * Defines the face alignment abstract interface and provides an InsightFace-based
 * implementation. Alignment uses similarity transform to map detected landmarks
 * to reference positions, producing a 112×112 normalized face for feature extraction.
 */
#ifndef FACE_RECOGNITION_FACE_ALIGNER_H
#define FACE_RECOGNITION_FACE_ALIGNER_H

#include "common/types.h"
#include <array>

namespace face_rec {

/**
 * @brief 人脸对齐抽象接口 / Face aligner abstract interface
 *
 * 所有对齐实现需继承此类，实现 Align 方法。输入为原始图像和 5 个关键点，
 * 输出为 112×112 的对齐后人脸 BGR 图像。
 * All aligners must inherit this class and implement Align(). Input is the
 * source image plus 5 landmarks, output is a 112×112 aligned face in BGR.
 */
class IFaceAligner {
public:
    virtual ~IFaceAligner() = default;
    /**
     * @brief 使用关键点对齐人脸，输出 112×112 的 BGR 图像
     *        Align face using landmarks, output 112×112 BGR image
     * @param src       原始图像 / source image
     * @param landmarks 5 个面部关键点（眼、鼻、嘴角） / 5 facial landmarks
     * @param dst_data  输出缓冲区（至少 112*112*3 字节） / output buffer (min 112*112*3 bytes)
     * @return true 成功 / success, false 参数无效 / invalid parameters
     */
    virtual bool Align(const Image& src, const std::array<Point, 5>& landmarks, uint8_t* dst_data) = 0;
};

/**
 * @brief InsightFace 风格的人脸对齐器
 *        InsightFace-style face aligner
 *
 * 使用 InsightFace 定义的 5 个参考关键点坐标（源于 ArcFace / InsightFace 标准）。
 * 通过最小二乘法求解相似变换（旋转、缩放、平移），
 * 再通过逆映射双线性插值生成对齐人脸。
 * Uses the 5 reference landmark coordinates from the InsightFace/ArcFace standard.
 * Solves a similarity transform (rotation, scale, translation) via least squares,
 * then generates the aligned face through inverse-mapping bilinear interpolation.
 */
class InsightFaceAligner : public IFaceAligner {
public:
    bool Align(const Image& src, const std::array<Point, 5>& landmarks, uint8_t* dst_data) override;

private:
    /**
     * @brief InsightFace/ArcFace 标准参考关键点（112×112 坐标系）
     *        Standard InsightFace/ArcFace reference landmarks (112×112 coordinate space)
     *
     * 坐标顺序：左眼、右眼、鼻尖、左嘴角、右嘴角
     * Order: left eye, right eye, nose tip, left mouth corner, right mouth corner
     * 来源 / Source: https://github.com/deepinsight/insightface
     */
    static constexpr float kRefPoints[5][2] = {
        {38.2946f, 51.6963f},   // 左眼中心 / left eye center
        {73.5318f, 51.5014f},   // 右眼中心 / right eye center
        {56.0252f, 71.7366f},   // 鼻尖 / nose tip
        {41.5493f, 92.3655f},   // 左嘴角 / left mouth corner
        {70.7299f, 92.2041f}    // 右嘴角 / right mouth corner
    };
};

} // namespace face_rec

#endif // FACE_RECOGNITION_FACE_ALIGNER_H
