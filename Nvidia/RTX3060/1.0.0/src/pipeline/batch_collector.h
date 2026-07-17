/**
 * @file batch_collector.h
 * @brief GPU Face Recognition — Phase-level batch collector.
 *        Collects frames from multiple streams into batches for GPU inference.
 * @module Pipeline Layer
 */

#ifndef GPU_FACE_RECOGNITION_BATCH_COLLECTOR_H
#define GPU_FACE_RECOGNITION_BATCH_COLLECTOR_H

#include "common/types.h"
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <future>

namespace face_rec {

/**
 * @struct BatchSlot
 * @brief A single frame slot in the batch collector.
 *        Pre-allocated pool to avoid steady-state heap allocation.
 */
struct BatchSlot {
    Image image;                    // Input image view
    int stream_id = -1;             // Source stream ID
    std::promise<std::vector<DetectedObject>> promise;
    std::future<std::vector<DetectedObject>> future;

    void Reset() {
        image = Image{};
        stream_id = -1;
        promise = std::promise<std::vector<DetectedObject>>{};
        future = promise.get_future();
    }
};

/**
 * @class BatchCollector
 * @brief Collects incoming frames and dispatches batched GPU inference.
 *        - Pre-allocated slot pool (max 16 slots)
 *        - 8ms adaptive timeout (1 stream → batch=1, full 16 → immediate)
 *        - Single GPU inference thread
 */
class BatchCollector {
public:
    /**
     * @brief Constructor
     * @param max_batch_size Maximum batch size (default 16)
     * @param timeout_ms Timeout before dispatching partial batch (default 8ms)
     */
    explicit BatchCollector(int max_batch_size = 16, int timeout_ms = 8);
    ~BatchCollector();

    /**
     * @brief Submit a frame for batched inference.
     * @param image Input image
     * @param stream_id Source stream identifier
     * @return Future containing the detection results
     */
    std::future<std::vector<DetectedObject>> Submit(
        const Image& image, int stream_id);

    /**
     * @brief Start the batch worker thread.
     * @param worker_fn Function to call with each collected batch
     */
    void Start(std::function<void(std::vector<BatchSlot>&)> worker_fn);

    /** @brief Stop the batch worker thread */
    void Stop();

    /** @brief Check if the collector is running */
    bool IsRunning() const { return running_.load(); }

private:
    int max_batch_size_;
    int timeout_ms_;
    std::vector<BatchSlot> slots_;
    std::mutex mutex_;
    std::condition_variable cv_;
    int current_count_ = 0;
    std::atomic<bool> running_{false};
    std::thread worker_thread_;
    std::function<void(std::vector<BatchSlot>&)> worker_fn_;

    void WorkerLoop();
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_BATCH_COLLECTOR_H
