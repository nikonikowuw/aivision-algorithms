/**
 * thread_pool.cpp
 *
 * 线程池模块实现 - Thread Pool Module Implementation
 *
 * 实现了工作线程的创建、任务循环和优雅关闭。
 * Implements worker thread creation, the task loop, and graceful shutdown.
 */

#include "thread_pool.h"

namespace face_rec {

/**
 * 构造函数：创建并启动指定数量的工作线程
 * Constructor: create and start the specified number of worker threads
 *
 * 每个工作线程执行一个无限循环：
 *   1. 获取互斥锁，等待条件变量（有新任务或 stop_ 标志）
 *   2. 如果 stop_ 且队列为空则退出
 *   3. 从队列中取出一个任务并执行
 *
 * Each worker runs an infinite loop:
 *   1. Acquire mutex, wait on condition variable (new task or stop_ flag)
 *   2. Exit if stop_ is set and the queue is empty
 *   3. Pop one task from the queue and execute it
 */
ThreadPool::ThreadPool(size_t num_threads) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(
            [this] {
                // 工作线程主循环 / Worker thread main loop
                for (;;) {
                    std::function<void()> task;
                    {
                        // 等待条件变量：有任务或收到停止信号
                        // Wait on condition variable: new task or stop signal
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock,
                            [this] { return this->stop_ || !this->tasks_.empty(); });

                        // 如果已停止且任务队列为空，线程退出
                        // If stopped and queue is empty, exit the thread
                        if (this->stop_ && this->tasks_.empty()) {
                            return;
                        }

                        // 从队列头部取出任务 / Pop a task from the front of the queue
                        task = std::move(this->tasks_.front());
                        this->tasks_.pop();
                    }

                    // 在锁外执行任务，避免长时间占锁
                    // Execute task outside the lock to avoid long-held mutex
                    task();
                }
            }
        );
    }
}

/**
 * 析构函数：优雅关闭线程池
 * Destructor: gracefully shut down the thread pool
 *
 * 流程 / Flow:
 *   1. 设置 stop_ 标志 / Set the stop_ flag
 *   2. 通知所有等待中的工作线程 / Notify all waiting worker threads
 *   3. 等待每个线程结束并回收 / Join each thread
 */
ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    // 唤醒所有阻塞在工作线程上的条件变量等待
    // Wake up all workers blocked on the condition variable
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

} // namespace face_rec
