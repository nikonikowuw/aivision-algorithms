/**
 * face_index.h
 *
 * 人脸底库索引模块 - Face Database Index Module
 *
 * 使用 Copy-on-Write 双缓冲（Snapshot）机制实现无锁读、串行写的人脸底库。
 * Implements a thread-safe face database using a Copy-on-Write double-buffer
 * (Snapshot) pattern: lock-free reads via shared_ptr, serialized writes via
 * atomic snapshot swap.
 *
 * 读线程通过 shared_ptr 持有当前快照，写线程拷贝→修改→原子替换。
 * Readers hold a shared_ptr to the current snapshot; writers copy, mutate,
 * then atomically swap it.
 *
 * 识别流程：
 *   1. Reader 调用 Read() 获取当前快照的 shared_ptr
 *   2. 对库中所有人脸计算余弦相似度
 *   3. 按 identity_id 去重（每个 ID 取最高分）
 *   4. 排序取 Top-K 返回
 *
 * Recognition flow:
 *   1. Reader calls Read() to obtain a shared_ptr to the current snapshot
 *   2. Compute cosine similarity against all identities in the database
 *   3. Deduplicate by identity_id (keep max score per ID)
 *   4. Sort and return Top-K results
 */

#ifndef FACE_RECOGNITION_FACE_INDEX_H
#define FACE_RECOGNITION_FACE_INDEX_H

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <functional>

namespace face_rec {

/**
 * KnownIdentity - 已知人脸身份 / A known face identity stored in the database
 */
struct KnownIdentity {
    std::string id;                    // 唯一身份标识 / Unique identity ID
    std::string name;                  // 显示名称 / Display name
    std::vector<float> embedding;      // 512 维归一化人脸特征向量 / 512-d normalized face embedding
};

/**
 * SearchResult - 检索结果单条记录 / A single search result entry
 *
 * 包含匹配的身份信息和相似度。identity_id / identity_name 通过值拷贝存储，
 * 不依赖外部快照生命周期，避免悬空指针。
 * Contains the matched identity info and similarity score. identity_id and
 * identity_name are stored by value, independent of the snapshot lifetime.
 */
struct SearchResult {
    float similarity;                 // 余弦相似度 / Cosine similarity score
    std::string identity_id;          // 匹配身份 ID / Matched identity ID
    std::string identity_name;        // 匹配身份名称 / Matched identity display name
};

/**
 * FaceIndex - 人脸底库索引（双缓冲快照模型）
 * Face database index (double-buffer snapshot model)
 *
 * 线程安全设计 / Thread safety design:
 *   - Read():  持有 current_mutex_ 返回 current_ 的 shared_ptr（读者可长时间持有）
 *              Hold current_mutex_ to return shared_ptr of current_; reader can hold it long
 *   - Update(): 先占 write_mutex_ 串行写，拷贝旧快照，执行修改，原子替换 current_
 *               Acquire write_mutex_ for serialization, copy old snapshot, mutate, atomically swap
 */
class FaceIndex {
public:
    /**
     * Snapshot - 不可变底库快照 / Immutable database snapshot
     *
     * 双缓冲的"读端"：读者通过 shared_ptr 持有此结构，保证读取期间数据不被修改。
     * The "read side" of the double buffer: readers hold a shared_ptr to this
     * struct, guaranteeing the data is not mutated during reading.
     */
    struct Snapshot {
        std::vector<KnownIdentity> identities;  // 全量身份列表 / Full identity list
        std::string version;                     // 底库版本号 / Database version string
    };

    FaceIndex() = default;
    ~FaceIndex() = default;

    /** 初始化底库，设置相似度阈值 / Initialize the index with a similarity threshold */
    void Initialize(float threshold);

    /**
     * Reader 接口：获取当前快照的共享指针
     * Reader API: obtain a shared_ptr to the current snapshot
     *
     * 返回的 shared_ptr 可被读者长时间持有，不影响 Writer 更新。
     * The returned shared_ptr can be held by the reader for a long time
     * without blocking writers.
     */
    std::shared_ptr<const Snapshot> Read() const;

    /**
     * Writer 接口：原子更新底库（拷贝 -> 修改 -> 交换）
     * Writer API: atomically update the database (copy -> mutate -> swap)
     *
     * mutator 是一个可调用对象，接收 Snapshot& 并对其进行修改。
     * The mutator is a callable that receives a Snapshot& and mutates it.
     */
    void Update(std::function<void(Snapshot&)> mutator);

    /**
     * 余弦相似度 Top-K 搜索
     * Cosine similarity Top-K search
     *
     * 按 identity_id 去重，每个 ID 保留最高相似度，降序排序后取前 K 个。
     * Deduplicates by identity_id (keeps the highest score per ID),
     * sorts descending and returns at most K results.
     */
    std::vector<SearchResult> SearchTopK(
        const float* query, size_t dim, int k = 5) const;

private:
    /** 当前活跃快照（不可变） / Current active snapshot (immutable) */
    std::shared_ptr<const Snapshot> current_ = std::make_shared<Snapshot>();
    /** 保护 current_ 读访问的互斥锁 / Mutex protecting read access to current_ */
    mutable std::mutex current_mutex_;
    /** 串行化写操作的互斥锁 / Mutex serializing write operations */
    std::mutex write_mutex_;
    /** 相似度阈值：低于此值的匹配被过滤 / Similarity threshold */
    float threshold_ = 0.45f;
};

} // namespace face_rec

#endif // FACE_RECOGNITION_FACE_INDEX_H
