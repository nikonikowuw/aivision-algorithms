// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Pipeline orchestrator.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../common/config.h"
#include "../common/types.h"
#include "../event/event_state.h"
#include "../models/damoyolo_detector.h"
#include "../preprocess/metal_preprocessor.h"
#include "../tracking/tracker.h"

namespace smoking {

class SmokingPipeline {
public:
    SmokingPipeline();
    ~SmokingPipeline();

    ErrorCode Initialize(const AlgoConfig& config, std::string& error);
    ErrorCode Infer(const uint8_t* frame_data, int32_t frame_width,
                    int32_t frame_height, uint32_t pixel_format,
                    int32_t stride, std::vector<SmokingEvent>& events,
                    std::string& error);
    ErrorCode InferNative(uint64_t native_handle, uint32_t pixel_format,
                          int32_t frame_width, int32_t frame_height,
                          std::vector<SmokingEvent>& events,
                          std::string& error);
    void Destroy();

    static const char* Version();
    static const char* Name();

private:
    struct FrameInput {
        const uint8_t* bgr_data = nullptr;
        uint64_t native_handle = 0;
        int32_t width = 0;
        int32_t height = 0;
        uint32_t pixel_format = kPixelFormatBGR24;
        int32_t stride = 0;

        bool IsNative() const { return native_handle != 0; }
    };

    struct PersonRoi {
        int32_t track_id = -1;
        RectF roi;
    };

    AlgoConfig config_;
    bool initialized_ = false;
    std::unique_ptr<DamoYoloDetector> person_detector_;
    std::unique_ptr<DamoYoloDetector> cigarette_detector_;
    std::unique_ptr<MetalPreprocessor> preprocessor_;
    std::unique_ptr<ITracker> tracker_;
    std::unique_ptr<EventStateMachine> event_sm_;
    int64_t last_analysis_time_ms_ = 0;
    int64_t analysis_interval_ms_ = 250;
    size_t tile_cursor_ = 0;

    ErrorCode Analyze(const FrameInput& frame,
                      std::vector<SmokingEvent>& events,
                      std::string& error);
    ErrorCode DetectRegion(DamoYoloDetector& detector,
                           const FrameInput& frame, const RectF& region,
                           std::vector<Detection>& detections,
                           std::string& error);
    std::vector<RectF> GenerateTiles(int32_t frame_width,
                                     int32_t frame_height) const;
    std::vector<PersonRoi> BuildUpperBodyRois(
        const std::vector<TrackedPerson>& persons, int32_t frame_width,
        int32_t frame_height) const;
    ErrorCode RunCigaretteDetection(
        const FrameInput& frame, const std::vector<PersonRoi>& rois,
        std::vector<SmokingEvidence>& evidences, std::string& error);
    std::vector<SmokingEvidence> AssociateCigarettes(
        const std::vector<SmokingEvidence>& evidences,
        const std::vector<TrackedPerson>& persons,
        const std::vector<PersonRoi>& rois) const;
    std::vector<SmokingEvent> ProcessEvents(
        const std::vector<TrackedPerson>& persons,
        const std::vector<SmokingEvidence>& evidences,
        const std::vector<PersonRoi>& evaluated_rois,
        int64_t current_time_ms);
    bool IsInsideConfiguredRegion(const RectF& bbox, int32_t frame_width,
                                  int32_t frame_height) const;
};

}  // namespace smoking
