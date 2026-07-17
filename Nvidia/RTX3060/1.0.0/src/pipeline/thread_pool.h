/**
 * @file thread_pool.h
 * @brief GPU Face Recognition — Simple thread pool for CPU tasks.
 * @module Pipeline Layer
 */

#ifndef GPU_FACE_RECOGNITION_THREAD_POOL_H
#define GPU_FACE_RECOGNITION_THREAD_POOL_H

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>

namespace face_rec {

/**
 * @class ThreadPool
 * @brief Simple thread pool for parallel CPU tasks (FAISS search, postprocessing).
 */
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    /**
     * @brief Submit a task and get a future for the result.
     */
    template<typename F, typename... Args>
    auto Submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))>;

    /** @brief Wait for all tasks to complete */
    void WaitForAll();

    /** @brief Get number of pending tasks */
    size_t PendingTasks() const;

    /** @brief Stop the pool and join all threads */
    void Shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
};

// Template implementation must be in header
template<typename F, typename... Args>
auto ThreadPool::Submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
    using return_type = decltype(f(args...));
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    auto future = task->get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (stop_) return std::future<return_type>();
        tasks_.emplace([this, task]() {
            active_tasks_++;
            (*task)();
            active_tasks_--;
            finished_.notify_all();
        });
    }
    condition_.notify_one();
    return future;
}

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_THREAD_POOL_H
