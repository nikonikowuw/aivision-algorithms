/**
 * face_index.cpp
 *
 * 人脸底库索引模块实现 - Face Database Index Module Implementation
 *
 * 实现了双缓冲快照的原子切换、余弦相似度 Top-K 搜索（含 identity_id 去重）。
 * Implements atomic snapshot swap for the double-buffer pattern,
 * cosine similarity Top-K search with identity_id deduplication.
 */

#include "face_index.h"
#include <unordered_map>
#include <algorithm>

namespace face_rec {

/**
 * 初始化底库：创建初始快照，设置版本号为 "initial"
 * Initialize the index: create the initial snapshot with version "initial"
 */
void FaceIndex::Initialize(float threshold) {
    threshold_ = threshold;
    std::lock_guard<std::mutex> lock(write_mutex_);
    auto new_snapshot = std::make_shared<Snapshot>();
    new_snapshot->version = "initial";
    std::atomic_store(&current_, std::shared_ptr<const Snapshot>(std::move(new_snapshot)));
}

/**
 * Reader 接口：返回当前快照的 shared_ptr
 * Reader API: return a shared_ptr to the current snapshot
 *
 * 注意：本函数只短暂持有互斥锁来读取 shared_ptr，然后立即返回给调用者。
 * 调用者可以安全地持有该指针，即使 Writer 后续更新了底库。
 * Note: the mutex is only held briefly to read the shared_ptr, then the
 * pointer is returned to the caller. The caller can safely hold the snapshot
 * even if a Writer updates the index later.
 */
std::shared_ptr<const FaceIndex::Snapshot> FaceIndex::Read() const {
    return std::atomic_load(&current_);
}

/**
 * Writer 接口：Copy-on-Write 原子更新底库
 * Writer API: Copy-on-Write atomic update
 *
 * 流程 / Flow:
 *   1. 拷贝当前快照 / Copy the current snapshot
 *   2. 调用 mutator 修改拷贝 / Apply mutator to the copy
 *   3. 原子替换 current_ / Atomically swap current_
 */
void FaceIndex::Update(std::function<void(Snapshot&)> mutator) {
    // 串行化写操作 / Serialize write operations
    std::lock_guard<std::mutex> write_lock(write_mutex_);

    // 深拷贝当前快照 / Deep-copy the current snapshot
    auto new_snapshot = std::make_shared<Snapshot>(*Read());

    // 执行调用者提供的修改逻辑 / Execute the user-provided mutation
    mutator(*new_snapshot);

    std::atomic_store(&current_, std::shared_ptr<const Snapshot>(std::move(new_snapshot)));
}

void FaceIndex::Replace(Snapshot snapshot) {
    std::lock_guard<std::mutex> write_lock(write_mutex_);
    auto new_snapshot = std::make_shared<Snapshot>(std::move(snapshot));
    std::atomic_store(&current_, std::shared_ptr<const Snapshot>(std::move(new_snapshot)));
}

/**
 * 余弦相似度 Top-K 搜索
 * Cosine similarity Top-K search
 *
 * 详细流程 / Detailed flow:
 *   1. 获取当前快照 / Acquire current snapshot
 *   2. 遍历所有人脸，计算 L2 归一化后的点积（等价于余弦相似度）
 *      Iterate all identities, compute dot product on L2-normalized vectors (=cosine similarity)
 *   3. 用 unordered_map 按 identity_id 去重，保留最高分
 *      Use unordered_map to deduplicate by identity_id, keeping the max score
 *   4. 收集结果，降序排序 / Collect results, sort descending
 *   5. 仅保留 Top-K / Keep only top-K
 */
std::vector<SearchResult> FaceIndex::SearchTopK(
    const float* query, size_t dim, int k) const {

    std::vector<SearchResult> results;
    if (!query || dim == 0) return results;

    // 获取当前不可变快照 / Get the current immutable snapshot
    auto snapshot = Read();
    if (snapshot->identities.empty()) return results;

    // 用哈希表按 identity_id 去重，只保留每条记录的最高相似度
    // Deduplicate by identity_id: keep the best similarity for each ID
    std::unordered_map<std::string, SearchResult> best_matches;
    best_matches.reserve(snapshot->identities.size());

    const bool has_contiguous_embeddings =
        snapshot->embedding_dim == dim &&
        snapshot->embeddings.size() == snapshot->identities.size() * dim;

    for (const auto& identity : snapshot->identities) {
        // 维度不匹配则跳过 / Skip if dimension mismatch
        if (!has_contiguous_embeddings && identity.embedding.size() != dim) continue;

        // 计算 L2 归一化后的点积（预归一化特征，点积 = 余弦相似度）
        // Compute dot product on pre-L2-normalized embeddings (= cosine similarity)
        float similarity = 0.0f;
        const size_t identity_index = static_cast<size_t>(&identity - snapshot->identities.data());
        const float* embedding = has_contiguous_embeddings
            ? snapshot->embeddings.data() + identity_index * dim
            : identity.embedding.data();
        for (size_t i = 0; i < dim; ++i) {
            similarity += query[i] * embedding[i];
        }

        // 低于阈值的过滤掉 / Filter out low-similarity matches
        if (similarity < threshold_) continue;

        // 按 ID 去重，只保留最高分 / Dedup by ID, keep only the best score
        auto it = best_matches.find(identity.id);
        if (it == best_matches.end() || similarity > it->second.similarity) {
            best_matches[identity.id] = SearchResult{similarity, identity.id, identity.name};
        }
    }

    // 收集去重后的结果 / Collect deduplicated results
    for (const auto& pair : best_matches) {
        results.push_back(pair.second);
    }

    if (k <= 0) return {};
    const size_t top_count = std::min(results.size(), static_cast<size_t>(k));
    std::partial_sort(results.begin(), results.begin() + top_count, results.end(),
                      [](const SearchResult& a, const SearchResult& b) {
                          return a.similarity > b.similarity;
                      });
    results.resize(top_count);

    return results;
}

} // namespace face_rec
