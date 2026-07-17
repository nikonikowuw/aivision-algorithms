/**
 * @file timer.h
 * @brief High-resolution wall-clock timer for inference latency measurement.
 *
 * Uses std::chrono::high_resolution_clock (typically std::chrono::steady_clock
 * on Linux, mach_absolute_time on macOS). Provides microsecond precision.
 *
 * ## Overflow
 *
 * The return type is uint32_t microseconds. At maximum value (2³²−1 µs),
 * this represents ~71.6 minutes of elapsed time — far exceeding any
 * realistic single-frame inference duration.
 *
 * ## Usage
 *
 *   Timer t;
 *   t.start();
 *   // ... inference ...
 *   uint32_t us = t.elapsed_us();
 */

#ifndef SAFETY_HELMET_TIMER_H
#define SAFETY_HELMET_TIMER_H

#include <chrono>

namespace safety_helmet {

class Timer {
public:
    Timer() = default;

    /**
     * @brief Record the current time as the start point.
     *
     * Calling start() again resets the timer — the previous start time is lost.
     */
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief Return elapsed microseconds since the last start() call.
     *
     * @return Microseconds as uint32_t. Wraps after ~71.6 minutes.
     */
    uint32_t elapsed_us() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_TIMER_H
