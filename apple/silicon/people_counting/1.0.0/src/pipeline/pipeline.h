/**
 * @file pipeline.h
 * @brief People Counting pipeline orchestrator
 */

#ifndef PEOPLE_COUNTING_PIPELINE_H
#define PEOPLE_COUNTING_PIPELINE_H

#include "common/types.h"
#include "common/config.h"
#include "common/logger.h"
#include "models/yolov8_detector.h"
#include "pipeline/tracker.h"
#include "pipeline/counting.h"
#include "runtime/coreml_backend.h"
#include "algo/abi_contract.h"
#include <memory>
#include <string>
#include <vector>
#include <mutex>

namespace people_count {

class PipelineOrchestrator {
public:
    PipelineOrchestrator() = default;
    ~PipelineOrchestrator();

    bool Initialize(const char* config_json);
    int Infer(const hw_buffer_desc_t* input, const char* context_json,
              infer_result_t* result);
    void Destroy();

private:
    Config config_;
    bool initialized_ = false;

    std::shared_ptr<CoreMLBackend> backend_;
    std::unique_ptr<YOLOv8Detector> detector_;
    std::unique_ptr<ITracker> tracker_;
    std::unique_ptr<LineCounter> counter_;

    int count_in_ = 0;
    int count_out_ = 0;
    int frame_count_ = 0;

    std::mutex infer_mutex_;
    PipelineTiming timing_;
    std::vector<uint8_t> decoded_bgr_buffer_;
};

} // namespace people_count

#endif // PEOPLE_COUNTING_PIPELINE_H
