/**
 * @file batch_collector.cpp
 * @brief GPU Face Recognition — BatchCollector implementation.
 */

#include "batch_collector.h"
#include "common/logger.h"
#include <thread>
#include <chrono>

namespace face_rec {

BatchCollector::BatchCollector(int max_batch_size, int timeout_ms)
    : max_batch_size_(max_batch_size), timeout_ms_(timeout_ms) {
    slots_.resize(max_batch_size);
    for (auto& slot : slots_) {
        slot.Reset();
    }
}

BatchCollector::~BatchCollector() {
    Stop();
}

void BatchCollector::Start(std::function<void(std::vector<BatchSlot>&)> worker_fn) {
    if (running_.load()) return;
    worker_fn_ = std::move(worker_fn);
    running_.store(true);
    worker_thread_ = std::thread(&BatchCollector::WorkerLoop, this);
    ALGO_LOGI(PIPELINE, "BatchCollector started (max_batch=%d, timeout=%dms)",
              max_batch_size_, timeout_ms_);
}

void BatchCollector::Stop() {
    running_.store(false);
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    ALGO_LOGI(PIPELINE, "BatchCollector stopped");
}

std::future<std::vector<DetectedObject>> BatchCollector::Submit(
    const Image& image, int stream_id) {

    std::unique_lock<std::mutex> lock(mutex_);

    if (current_count_ >= max_batch_size_) {
        // Batch full — wait for next cycle
        cv_.wait(lock, [this] { return current_count_ < max_batch_size_; });
    }

    int slot_idx = current_count_++;
    slots_[slot_idx].image = image;
    slots_[slot_idx].stream_id = stream_id;
    slots_[slot_idx].promise = std::promise<std::vector<DetectedObject>>{};

    auto future = slots_[slot_idx].promise.get_future();

    // If batch is full, wake the worker immediately
    if (current_count_ >= max_batch_size_) {
        cv_.notify_all();
    }

    return future;
}

void BatchCollector::WorkerLoop() {
    while (running_.load()) {
        std::vector<BatchSlot> ready_batch;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for at least one slot
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms_),
                         [this] { return current_count_ > 0; });

            // Collect all pending slots
            if (current_count_ > 0) {
                ready_batch.resize(current_count_);
                for (int i = 0; i < current_count_; ++i) {
                    ready_batch[i] = std::move(slots_[i]);
                    slots_[i].Reset();
                }
                current_count_ = 0;
            }
        }

        if (!ready_batch.empty() && worker_fn_) {
            ALGO_LOGD(PIPELINE, "Dispatching batch of %zu frames", ready_batch.size());
            worker_fn_(ready_batch);

            // Set results via promises
            // (worker_fn_ is expected to set the results on the promises)
        }
    }
}

} // namespace face_rec
