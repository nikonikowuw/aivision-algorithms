// Copyright (c) 2024. All rights reserved.
// Smoking Detection Algorithm - Pipeline orchestrator implementation.

#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../common/geometry_utils.h"
#include "../common/logger.h"
#include "../postprocess/nms.h"
#include "../runtime/coreml_backend.h"

namespace smoking {
namespace {

bool IsNonEmptyDirectory(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_directory(path, error) || error) return false;
    for (std::filesystem::recursive_directory_iterator it(path, error), end;
         !error && it != end; it.increment(error)) {
        if (it->is_regular_file(error) && !error && it->file_size(error) > 0) {
            return true;
        }
    }
    return false;
}

int64_t MonotonicMilliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

SmokingPipeline::SmokingPipeline() = default;
SmokingPipeline::~SmokingPipeline() { Destroy(); }

ErrorCode SmokingPipeline::Initialize(const AlgoConfig& config,
                                      std::string& error) {
    Destroy();
    config_ = config;
    error.clear();

    if (ValidateConfig(config_, error) != ErrorCode::kSuccess) {
        return ErrorCode::kInvalidParam;
    }
    if (config_.runtime_root.empty()) {
        error = "runtime_root was not supplied by the ABI layer";
        return ErrorCode::kInvalidParam;
    }

    const std::filesystem::path runtime_root(config_.runtime_root);
    std::filesystem::path model_dir(config_.model_dir);
    if (model_dir.is_relative()) model_dir = runtime_root / model_dir;
    const auto human_model = model_dir / "damoyolo_human.mlmodelc";
    const auto cigarette_model = model_dir / "damoyolo_cigarette.mlmodelc";
    const auto metallib = runtime_root / "shaders" / "image_preprocess.metallib";
    std::error_code filesystem_error;
    if (!IsNonEmptyDirectory(human_model)) {
        error = "human CoreML model bundle is missing or empty: " +
                human_model.string();
        return ErrorCode::kInferenceFailed;
    }
    if (!IsNonEmptyDirectory(cigarette_model)) {
        error = "cigarette CoreML model bundle is missing or empty: " +
                cigarette_model.string();
        return ErrorCode::kInferenceFailed;
    }
    if (!std::filesystem::is_regular_file(metallib, filesystem_error) ||
        filesystem_error || std::filesystem::file_size(metallib, filesystem_error) == 0 ||
        filesystem_error) {
        error = "Metal shader library is missing or empty: " + metallib.string();
        return ErrorCode::kPreprocessFailed;
    }

    SetGlobalLogLevel(static_cast<LogLevel>(config_.log_level));
    analysis_interval_ms_ = std::max<int64_t>(
        1, static_cast<int64_t>(1000.0 / config_.analysis_fps));

    preprocessor_ = std::make_unique<MetalPreprocessor>();
    if (!preprocessor_->Initialize(640, 640, metallib.string(), error)) {
        Destroy();
        return ErrorCode::kPreprocessFailed;
    }

    ModelSpec human_spec;
    human_spec.model_path = human_model.string();
    human_spec.model_type = "human";
    human_spec.conf_threshold = config_.person_conf_threshold;
    human_spec.nms_threshold = config_.person_nms_threshold;
    person_detector_ = std::make_unique<DamoYoloDetector>(
        std::make_unique<CoreMLNativeBackend>());
    if (!person_detector_->Load(human_spec, error)) {
        Destroy();
        return ErrorCode::kInferenceFailed;
    }

    ModelSpec cigarette_spec;
    cigarette_spec.model_path = cigarette_model.string();
    cigarette_spec.model_type = "cigarette";
    cigarette_spec.conf_threshold = config_.cigarette_conf_threshold;
    cigarette_spec.nms_threshold = config_.cigarette_nms_threshold;
    cigarette_detector_ = std::make_unique<DamoYoloDetector>(
        std::make_unique<CoreMLNativeBackend>());
    if (!cigarette_detector_->Load(cigarette_spec, error)) {
        Destroy();
        return ErrorCode::kInferenceFailed;
    }

    TrackerConfig tracker_config;
    tracker_config.iou_threshold = config_.tracker_iou_threshold;
    tracker_config.max_lost_ms = config_.tracker_max_lost_ms;
    tracker_ = std::make_unique<PersonTracker>(tracker_config);

    EventStateMachine::Config event_config;
    event_config.temporal_window = config_.temporal_window;
    event_config.confirm_hits = config_.confirm_hits;
    event_config.rearm_ms = config_.rearm_ms;
    event_sm_ = std::make_unique<EventStateMachine>(event_config);

    last_analysis_time_ms_ = 0;
    tile_cursor_ = 0;
    initialized_ = true;
    return ErrorCode::kSuccess;
}

ErrorCode SmokingPipeline::Infer(const uint8_t* frame_data,
                                 int32_t frame_width, int32_t frame_height,
                                 uint32_t pixel_format, int32_t stride,
                                 std::vector<SmokingEvent>& events,
                                 std::string& error) {
    if (!initialized_ || !frame_data || pixel_format != kPixelFormatBGR24) {
        error = "pipeline is not initialized or BGR24 input is invalid";
        return ErrorCode::kInvalidParam;
    }
    FrameInput frame;
    frame.bgr_data = frame_data;
    frame.width = frame_width;
    frame.height = frame_height;
    frame.pixel_format = pixel_format;
    frame.stride = stride;
    return Analyze(frame, events, error);
}

ErrorCode SmokingPipeline::InferNative(uint64_t native_handle,
                                       uint32_t pixel_format,
                                       int32_t frame_width,
                                       int32_t frame_height,
                                       std::vector<SmokingEvent>& events,
                                       std::string& error) {
    if (!initialized_ || native_handle == 0) {
        error = "pipeline is not initialized or native input is null";
        return ErrorCode::kInvalidParam;
    }
    FrameInput frame;
    frame.native_handle = native_handle;
    frame.width = frame_width;
    frame.height = frame_height;
    frame.pixel_format = pixel_format;
    return Analyze(frame, events, error);
}

ErrorCode SmokingPipeline::Analyze(const FrameInput& frame,
                                   std::vector<SmokingEvent>& events,
                                   std::string& error) {
    events.clear();
    error.clear();
    if (frame.width <= 0 || frame.height <= 0) {
        error = "frame dimensions must be positive";
        return ErrorCode::kInvalidBuffer;
    }

    const int64_t current_time_ms = MonotonicMilliseconds();
    if (last_analysis_time_ms_ != 0 &&
        current_time_ms - last_analysis_time_ms_ < analysis_interval_ms_) {
        return ErrorCode::kSuccess;
    }
    last_analysis_time_ms_ = current_time_ms;

    std::vector<Detection> persons;
    ErrorCode status = DetectRegion(*person_detector_, frame,
                                    {0.0f, 0.0f,
                                     static_cast<float>(frame.width),
                                     static_cast<float>(frame.height)},
                                    persons, error);
    if (status != ErrorCode::kSuccess) return status;

    if (config_.tile_enabled) {
        const auto tiles = GenerateTiles(frame.width, frame.height);
        const size_t budget = std::min<size_t>(config_.tile_budget_per_tick,
                                               tiles.size());
        for (size_t offset = 0; offset < budget; ++offset) {
            const size_t index = (tile_cursor_ + offset) % tiles.size();
            std::vector<Detection> tile_detections;
            status = DetectRegion(*person_detector_, frame, tiles[index],
                                  tile_detections, error);
            if (status != ErrorCode::kSuccess) return status;
            persons.insert(persons.end(), tile_detections.begin(),
                           tile_detections.end());
        }
        if (!tiles.empty()) tile_cursor_ = (tile_cursor_ + budget) % tiles.size();
    }

    persons.erase(std::remove_if(persons.begin(), persons.end(),
                                 [&](const Detection& detection) {
                                     return !IsInsideConfiguredRegion(
                                         detection.bbox, frame.width,
                                         frame.height);
                                 }),
                  persons.end());
    SortByConfidence(persons);
    const auto person_nms = ApplyNMS(persons, config_.person_nms_threshold);
    std::vector<Detection> merged_persons;
    merged_persons.reserve(person_nms.kept_indices.size());
    for (const int32_t index : person_nms.kept_indices) {
        merged_persons.push_back(persons[static_cast<size_t>(index)]);
    }
    if (static_cast<int32_t>(merged_persons.size()) > config_.max_persons) {
        merged_persons.resize(static_cast<size_t>(config_.max_persons));
    }

    const auto tracked_persons = tracker_->Update(merged_persons,
                                                   current_time_ms);
    const auto rois = BuildUpperBodyRois(tracked_persons, frame.width,
                                         frame.height);
    std::vector<SmokingEvidence> cigarette_evidences;
    status = RunCigaretteDetection(frame, rois, cigarette_evidences, error);
    if (status != ErrorCode::kSuccess) return status;
    const auto associated = AssociateCigarettes(cigarette_evidences,
                                                 tracked_persons, rois);
    events = ProcessEvents(tracked_persons, associated, rois,
                           current_time_ms);

    std::vector<int32_t> active_track_ids;
    for (const auto& track : tracker_->GetActiveTracks()) {
        active_track_ids.push_back(track.track_id);
    }
    event_sm_->RetireMissingTracks(active_track_ids, current_time_ms);
    event_sm_->CleanupTombstones(current_time_ms);
    return ErrorCode::kSuccess;
}

ErrorCode SmokingPipeline::DetectRegion(DamoYoloDetector& detector,
                                        const FrameInput& frame,
                                        const RectF& region,
                                        std::vector<Detection>& detections,
                                        std::string& error) {
    PreparedImage prepared;
    const bool prepared_ok = frame.IsNative()
        ? preprocessor_->ProcessNative(frame.native_handle, frame.width,
                                       frame.height, frame.pixel_format,
                                       region, prepared, error)
        : preprocessor_->ProcessBGR24(frame.bgr_data, frame.width,
                                      frame.height, frame.stride, region,
                                      prepared, error);
    if (!prepared_ok) return ErrorCode::kPreprocessFailed;
    return detector.Detect(prepared.pixel_buffer, prepared.transform,
                           detections, error);
}

std::vector<RectF> SmokingPipeline::GenerateTiles(int32_t frame_width,
                                                  int32_t frame_height) const {
    std::vector<RectF> tiles;
    const float tile_width = frame_width /
        (config_.tile_grid_cols -
         config_.tile_overlap * (config_.tile_grid_cols - 1));
    const float tile_height = frame_height /
        (config_.tile_grid_rows -
         config_.tile_overlap * (config_.tile_grid_rows - 1));
    const float step_x = tile_width * (1.0f - config_.tile_overlap);
    const float step_y = tile_height * (1.0f - config_.tile_overlap);
    for (int32_t row = 0; row < config_.tile_grid_rows; ++row) {
        for (int32_t column = 0; column < config_.tile_grid_cols; ++column) {
            RectF tile{column * step_x, row * step_y,
                       tile_width, tile_height};
            tile.ClipToFrame(static_cast<float>(frame_width),
                             static_cast<float>(frame_height));
            tiles.push_back(tile);
        }
    }
    return tiles;
}

std::vector<SmokingPipeline::PersonRoi> SmokingPipeline::BuildUpperBodyRois(
    const std::vector<TrackedPerson>& persons, int32_t frame_width,
    int32_t frame_height) const {
    std::vector<PersonRoi> rois;
    rois.reserve(persons.size());
    for (const auto& person : persons) {
        const float expand_x = person.bbox.width * config_.roi_expand_x;
        const float expand_top = person.bbox.height * config_.roi_expand_top;
        RectF roi{person.bbox.x - expand_x,
                  person.bbox.y - expand_top,
                  person.bbox.width + 2.0f * expand_x,
                  person.bbox.height * config_.upper_body_ratio + expand_top};
        roi.ClipToFrame(static_cast<float>(frame_width),
                        static_cast<float>(frame_height));
        if (roi.width >= 10.0f && roi.height >= 10.0f) {
            rois.push_back({person.track_id, roi});
        }
    }
    return rois;
}

ErrorCode SmokingPipeline::RunCigaretteDetection(
    const FrameInput& frame, const std::vector<PersonRoi>& rois,
    std::vector<SmokingEvidence>& evidences, std::string& error) {
    evidences.clear();
    for (const auto& person_roi : rois) {
        std::vector<Detection> detections;
        const ErrorCode status = DetectRegion(*cigarette_detector_, frame,
                                              person_roi.roi, detections,
                                              error);
        if (status != ErrorCode::kSuccess) return status;
        for (const auto& detection : detections) {
            evidences.push_back({detection.bbox, detection.confidence, -1});
        }
    }

    std::vector<Detection> dedup_input;
    dedup_input.reserve(evidences.size());
    for (const auto& evidence : evidences) {
        dedup_input.push_back({evidence.cigarette_bbox,
                               evidence.confidence, 0});
    }
    SortByConfidence(dedup_input);
    const auto nms = ApplyNMS(dedup_input, config_.cigarette_nms_threshold);
    evidences.clear();
    for (const int32_t index : nms.kept_indices) {
        const auto& detection = dedup_input[static_cast<size_t>(index)];
        evidences.push_back({detection.bbox, detection.confidence, -1});
    }
    return ErrorCode::kSuccess;
}

std::vector<SmokingEvidence> SmokingPipeline::AssociateCigarettes(
    const std::vector<SmokingEvidence>& evidences,
    const std::vector<TrackedPerson>& persons,
    const std::vector<PersonRoi>& rois) const {
    struct Edge {
        size_t evidence = 0;
        size_t person = 0;
        float score = 0.0f;
    };
    std::unordered_map<int32_t, RectF> roi_by_track;
    for (const auto& roi : rois) roi_by_track[roi.track_id] = roi.roi;

    std::vector<Edge> edges;
    for (size_t evidence_index = 0; evidence_index < evidences.size();
         ++evidence_index) {
        for (size_t person_index = 0; person_index < persons.size();
             ++person_index) {
            const auto roi_it = roi_by_track.find(persons[person_index].track_id);
            if (roi_it == roi_by_track.end()) continue;
            const PointF center = evidences[evidence_index].cigarette_bbox.Center();
            const RectF& roi = roi_it->second;
            if (center.x < roi.Left() || center.x > roi.Right() ||
                center.y < roi.Top() || center.y > roi.Bottom()) {
                continue;
            }
            const float distance = std::min(
                1.0f, CenterDistanceNormalized(
                          evidences[evidence_index].cigarette_bbox, roi));
            const float score = evidences[evidence_index].confidence * 0.6f +
                                persons[person_index].confidence * 0.2f +
                                (1.0f - distance) * 0.2f;
            edges.push_back({evidence_index, person_index, score});
        }
    }
    std::sort(edges.begin(), edges.end(),
              [](const Edge& left, const Edge& right) {
                  return left.score > right.score;
              });

    std::vector<bool> used_evidence(evidences.size(), false);
    std::vector<bool> used_person(persons.size(), false);
    std::vector<SmokingEvidence> associated;
    for (const auto& edge : edges) {
        if (used_evidence[edge.evidence] || used_person[edge.person]) continue;
        SmokingEvidence evidence = evidences[edge.evidence];
        evidence.track_id = persons[edge.person].track_id;
        associated.push_back(evidence);
        used_evidence[edge.evidence] = true;
        used_person[edge.person] = true;
    }
    return associated;
}

std::vector<SmokingEvent> SmokingPipeline::ProcessEvents(
    const std::vector<TrackedPerson>& persons,
    const std::vector<SmokingEvidence>& evidences,
    const std::vector<PersonRoi>& evaluated_rois, int64_t current_time_ms) {
    std::unordered_map<int32_t, SmokingEvidence> evidence_by_track;
    for (const auto& evidence : evidences) {
        evidence_by_track[evidence.track_id] = evidence;
    }
    std::unordered_set<int32_t> evaluated;
    for (const auto& roi : evaluated_rois) evaluated.insert(roi.track_id);

    std::vector<SmokingEvent> events;
    for (const auto& person : persons) {
        if (evaluated.count(person.track_id) == 0) continue;
        const auto evidence = evidence_by_track.find(person.track_id);
        const bool has_evidence = evidence != evidence_by_track.end();
        const RectF cigarette_bbox = has_evidence
            ? evidence->second.cigarette_bbox : RectF{};
        const float cigarette_confidence = has_evidence
            ? evidence->second.confidence : 0.0f;
        SmokingEvent event = event_sm_->ProcessTrack(
            person.track_id, has_evidence, person.bbox, cigarette_bbox,
            cigarette_confidence, current_time_ms);
        if (event.track_id >= 0) {
            event.detect_confidence = std::min(person.confidence,
                                               cigarette_confidence);
            events.push_back(std::move(event));
        }
    }
    return events;
}

bool SmokingPipeline::IsInsideConfiguredRegion(const RectF& bbox,
                                                int32_t frame_width,
                                                int32_t frame_height) const {
    if (config_.detection_regions.empty()) return true;
    for (const auto& normalized_region : config_.detection_regions) {
        std::vector<PointF> pixels;
        pixels.reserve(normalized_region.points.size());
        for (const auto& point : normalized_region.points) {
            pixels.push_back({point.x * frame_width, point.y * frame_height});
        }
        if (RectIntersectsRegion(bbox, pixels)) return true;
    }
    return false;
}

void SmokingPipeline::Destroy() {
    initialized_ = false;
    event_sm_.reset();
    tracker_.reset();
    cigarette_detector_.reset();
    person_detector_.reset();
    if (preprocessor_) preprocessor_->Destroy();
    preprocessor_.reset();
    last_analysis_time_ms_ = 0;
    tile_cursor_ = 0;
}

const char* SmokingPipeline::Version() { return "1.0.0"; }
const char* SmokingPipeline::Name() { return "smoking_detection"; }

}  // namespace smoking
