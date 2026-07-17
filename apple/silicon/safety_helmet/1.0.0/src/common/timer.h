#ifndef SAFETY_HELMET_TIMER_H
#define SAFETY_HELMET_TIMER_H

#include <chrono>

namespace safety_helmet {

class Timer {
public:
    Timer() = default;

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    uint32_t elapsed_us() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace safety_helmet

#endif // SAFETY_HELMET_TIMER_H
