/**
 * thread_pool.h
 *
 * 线程池模块 - Thread Pool Module
 *
 * 实现了一个通用的任务队列线程池。
 * Implements a generic task-queue-based thread pool.
 *
 * Enqueue 模板方法支持任意可调用对象，返回 std::future 以获取执行结果。
 * The Enqueue template method accepts any callable and returns a std::future
 * for obtaining the result.
 *
 * 线程安全：任务队列受互斥锁保护，通过条件变量实现生产者-消费者通知。
 * Thread safety: the task queue is protected by a mutex; condition variables
 * are used for producer-consumer notification.
 */

#ifndef FACE_RECOGNITION_THREAD_POOL_H
#define FACE_RECOGNITION_THREAD_POOL_H

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <memory>
#include <stdexcept>

namespace face_rec {

/**
 * ThreadPool - 固定大小的工作线程池
 * A fixed-size worker thread pool
 *
 * 工作线程从共享队列中取出任务并执行。线程池析构时等待所有任务完成。
 * Worker threads pull tasks from the shared queue and execute them.
 * The pool waits for all tasks to finish upon destruction.
 */
class ThreadPool {
public:
    /**
     * 创建并启动 num_threads 个工作线程
     * Create and start num_threads worker threads
     */
    explicit ThreadPool(size_t num_threads);

    /** 析构：通知所有线程退出并等待它们结束 / Notify all threads to stop and join */
    ~ThreadPool();

    /**
     * 向队列添加一个异步任务，返回 std::future
     * Enqueue an asynchronous task and return a std::future
     *
     * 用法示例 / Usage example:
     *   auto fut = pool.Enqueue([](int a, int b) { return a + b; }, 1, 2);
     *   int result = fut.get();
     */
    template<class F, class... Args>
    auto Enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F, Args...>>;

    /** 返回工作线程数 / Return the number of worker threads */
    size_t Size() const { return workers_.size(); }

private:
    /** 工作线程集合 / Collection of worker threads */
    std::vector<std::thread> workers_;
    /** 任务队列（FIFO） / Task queue (FIFO) */
    std::queue<std::function<void()>> tasks_;

    /** 保护任务队列的互斥锁 / Mutex protecting the task queue */
    std::mutex queue_mutex_;
    /** 用于任务通知的条件变量 / Condition variable for task notification */
    std::condition_variable condition_;
    /** 线程池停止标志 / Stop flag */
    bool stop_ = false;
};

/**
 * Enqueue 模板实现：将任意可调用对象打包为 packaged_task 并加入队列
 * Template implementation: wrap any callable as a packaged_task and enqueue it
 *
 * @param f      可调用对象（函数、lambda、函数对象等） / Callable (function, lambda, functor, etc.)
 * @param args   调用参数 / Arguments to forward to the callable
 * @return       std::future<return_type> 用于获取结果 / Used to retrieve the result
 */
template<class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::invoke_result_t<F, Args...>> {
    using return_type = typename std::invoke_result_t<F, Args...>;

    // 将可调用对象 + 参数打包为 packaged_task / Wrap callable + args into a packaged_task
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );

    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);

        // 线程池已停止则拒绝新任务 / Reject new tasks if pool is stopped
        if (stop_) {
            throw std::runtime_error("Enqueue on stopped ThreadPool");
        }

        // 将任务放入队列 / Push task into the queue
        tasks_.emplace([task]() { (*task)(); });
    }
    // 通知一个等待的工作线程 / Notify one waiting worker thread
    condition_.notify_one();
    return res;
}

} // namespace face_rec

#endif // FACE_RECOGNITION_THREAD_POOL_H
