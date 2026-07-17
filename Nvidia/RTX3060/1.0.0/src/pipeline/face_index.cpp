/**
 * @file face_index.cpp
 * @brief GPU Face Recognition — Face index implementation.
 *        Short-term brute-force + FAISS gallery (when available).
 */

#include "face_index.h"
#include "common/logger.h"
#include <algorithm>
#include <cmath>

// FAISS include (conditional)
#ifdef HAS_FAISS
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#endif

namespace face_rec {

FaceIndex::FaceIndex() = default;

FaceIndex::~FaceIndex() {
#ifdef HAS_FAISS
    delete static_cast<faiss::Index*>(faiss_index_);
#endif
}

float FaceIndex::CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;  // Already L2-normalized
}

void FaceIndex::AddShortTerm(int track_id, const std::vector<float>& embedding,
                              const std::string& identity_id,
                              const std::string& identity_name) {
    std::lock_guard<std::mutex> lock(short_term_mutex_);
    ShortTermEntry entry;
    entry.embedding = embedding;
    entry.identity_id = identity_id;
    entry.identity_name = identity_name;
    short_term_[track_id] = entry;
}

bool FaceIndex::Search(const std::vector<float>& embedding, float threshold,
                       std::string& matched_id, std::string& matched_name,
                       float& similarity,
                       std::vector<DetectedObject::Candidate>& candidates) {
    candidates.clear();
    matched_id.clear();
    matched_name.clear();
    similarity = 0.0f;

    if (embedding.empty()) return false;

    // Stage 1: Search short-term store (brute-force)
    {
        std::lock_guard<std::mutex> lock(short_term_mutex_);
        for (const auto& [track_id, entry] : short_term_) {
            if (entry.embedding.empty()) continue;
            float sim = CosineSimilarity(embedding, entry.embedding);
            if (!entry.identity_id.empty() && sim > similarity) {
                similarity = sim;
                matched_id = entry.identity_id;
                matched_name = entry.identity_name;
            }
            if (sim > 0.1f) {
                DetectedObject::Candidate c;
                c.identity_id = entry.identity_id;
                c.identity_name = entry.identity_name;
                c.similarity = sim;
                candidates.push_back(c);
            }
        }
    }

    // Short-term hit → return early
    if (similarity >= threshold && !matched_id.empty()) {
        return true;
    }

    // Stage 2: Search FAISS gallery
    {
        std::lock_guard<std::mutex> lock(gallery_mutex_);
        if (!gallery_entries_.empty()) {
            constexpr int kTopK = 5;
            std::vector<size_t> indices;
            std::vector<float> distances;
            SearchGallery(embedding, kTopK, indices, distances);

            for (size_t i = 0; i < indices.size(); ++i) {
                float sim = 1.0f - distances[i];  // Convert L2 distance to similarity
                sim = std::max(0.0f, sim);
                const auto& entry = gallery_entries_[indices[i]];
                if (sim > similarity) {
                    similarity = sim;
                    matched_id = entry.identity_id;
                    matched_name = entry.identity_name;
                }
                DetectedObject::Candidate c;
                c.identity_id = entry.identity_id;
                c.identity_name = entry.identity_name;
                c.similarity = sim;
                candidates.push_back(c);
            }
        }
    }

    // Sort candidates by similarity descending
    std::sort(candidates.begin(), candidates.end(),
              [](const DetectedObject::Candidate& a, const DetectedObject::Candidate& b) {
                  return a.similarity > b.similarity;
              });
    if (candidates.size() > 5) {
        candidates.resize(5);
    }

    return similarity >= threshold && !matched_id.empty();
}

void FaceIndex::RebuildGallery(const std::vector<IdentityEntry>& entries) {
    std::lock_guard<std::mutex> lock(gallery_mutex_);
    gallery_entries_ = entries;
    BuildFaissIndex();
    ALGO_LOGI(FACE_INDEX, "Gallery rebuilt with %zu entries", entries.size());
}

void FaceIndex::AddToGallery(const IdentityEntry& entry) {
    std::lock_guard<std::mutex> lock(gallery_mutex_);
    gallery_entries_.push_back(entry);
    BuildFaissIndex();
}

size_t FaceIndex::GallerySize() const {
    std::lock_guard<std::mutex> lock(gallery_mutex_);
    return gallery_entries_.size();
}

void FaceIndex::ClearShortTerm() {
    std::lock_guard<std::mutex> lock(short_term_mutex_);
    short_term_.clear();
}

void FaceIndex::BuildFaissIndex() {
#ifdef HAS_FAISS
    // Delete old index
    delete static_cast<faiss::Index*>(faiss_index_);
    faiss_index_ = nullptr;

    if (gallery_entries_.empty()) return;

    constexpr int kDim = 512;
    size_t n = gallery_entries_.size();

    if (n < 1000) {
        // Small gallery: brute-force flat index
        auto* index = new faiss::IndexFlatIP(kDim);  // Inner product = cosine for L2-normalized
        std::vector<float> data(n * kDim);
        for (size_t i = 0; i < n; ++i) {
            std::copy(gallery_entries_[i].embedding.begin(),
                      gallery_entries_[i].embedding.end(),
                      data.begin() + i * kDim);
        }
        index->add(n, data.data());
        faiss_index_ = index;
        ALGO_LOGI(FACE_INDEX, "Built FAISS IndexFlatIP (%zu entries)", n);
    } else {
        // Larger gallery: IVFFlat
        int nlist = static_cast<int>(std::sqrt(n));
        auto* quantizer = new faiss::IndexFlatIP(kDim);
        auto* index = new faiss::IndexIVFFlat(quantizer, kDim, nlist, faiss::METRIC_INNER_PRODUCT);
        std::vector<float> data(n * kDim);
        for (size_t i = 0; i < n; ++i) {
            std::copy(gallery_entries_[i].embedding.begin(),
                      gallery_entries_[i].embedding.end(),
                      data.begin() + i * kDim);
        }
        index->train(n, data.data());
        index->add(n, data.data());
        index->nprobe = std::min(16, nlist);
        faiss_index_ = index;
        ALGO_LOGI(FACE_INDEX, "Built FAISS IndexIVFFlat (%zu entries, nlist=%d)", n, nlist);
    }
#else
    // Brute-force fallback when FAISS is not available
    ALGO_LOGW(FACE_INDEX, "FAISS not available — using brute-force gallery search");
#endif
}

void FaceIndex::SearchGallery(const std::vector<float>& embedding, int k,
                               std::vector<size_t>& indices,
                               std::vector<float>& distances) const {
    indices.clear();
    distances.clear();

    if (gallery_entries_.empty()) return;

#ifdef HAS_FAISS
    if (faiss_index_) {
        auto* index = static_cast<faiss::Index*>(faiss_index_);
        int actual_k = std::min(k, static_cast<int>(gallery_entries_.size()));
        std::vector<float> dists(actual_k);
        std::vector<faiss::Index::idx_t> ids(actual_k);
        index->search(1, embedding.data(), actual_k, dists.data(), ids.data());
        for (int i = 0; i < actual_k; ++i) {
            if (ids[i] >= 0) {
                indices.push_back(static_cast<size_t>(ids[i]));
                distances.push_back(dists[i]);
            }
        }
        return;
    }
#endif

    // Brute-force fallback
    struct ScoredIdx { float score; size_t idx; };
    std::vector<ScoredIdx> scored;
    scored.reserve(gallery_entries_.size());

    for (size_t i = 0; i < gallery_entries_.size(); ++i) {
        float sim = CosineSimilarity(embedding, gallery_entries_[i].embedding);
        scored.push_back({sim, i});
    }

    std::sort(scored.begin(), scored.end(),
              [](const ScoredIdx& a, const ScoredIdx& b) { return a.score > b.score; });

    int count = std::min(k, static_cast<int>(scored.size()));
    for (int i = 0; i < count; ++i) {
        indices.push_back(scored[i].idx);
        distances.push_back(1.0f - scored[i].score);  // Convert to L2-like distance
    }
}

} // namespace face_rec
