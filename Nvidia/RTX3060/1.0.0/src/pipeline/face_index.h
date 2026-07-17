/**
 * @file face_index.h
 * @brief GPU Face Recognition — Short-term store + FAISS gallery matching.
 * @module Pipeline Layer
 */

#ifndef GPU_FACE_RECOGNITION_FACE_INDEX_H
#define GPU_FACE_RECOGNITION_FACE_INDEX_H

#include "common/types.h"
#include <vector>
#include <string>
#include <mutex>
#include <unordered_map>

namespace face_rec {

/**
 * @struct IdentityEntry
 * @brief A single registered face entry in the gallery.
 */
struct IdentityEntry {
    std::string identity_id;
    std::string identity_name;
    std::vector<float> embedding;  // 512-d L2-normalized
};

/**
 * @class FaceIndex
 * @brief Face identity matching engine.
 *        Short-term: std::vector brute-force scan (10K entries, <1ms)
 *        Gallery: FAISS Index (IndexFlat / IVFFlat / IVFPQ parameterized by scale)
 */
class FaceIndex {
public:
    FaceIndex();
    ~FaceIndex();

    /**
     * @brief Search for the best match in short-term + gallery.
     * @param embedding 512-d normalized face embedding
     * @param threshold Cosine similarity threshold
     * @param[out] matched_id Matched identity ID (empty if no match)
     * @param[out] matched_name Matched identity name
     * @param[out] similarity Best similarity score
     * @param[out] candidates Top-K candidates
     * @return true if a match was found above threshold
     */
    bool Search(const std::vector<float>& embedding, float threshold,
                std::string& matched_id, std::string& matched_name,
                float& similarity,
                std::vector<DetectedObject::Candidate>& candidates);

    /**
     * @brief Add an embedding to the short-term store (from a tracked person).
     */
    void AddShortTerm(int track_id, const std::vector<float>& embedding,
                      const std::string& identity_id = "",
                      const std::string& identity_name = "");

    /**
     * @brief Rebuild the FAISS gallery from a full JSON snapshot.
     * @param entries Vector of identity entries
     */
    void RebuildGallery(const std::vector<IdentityEntry>& entries);

    /**
     * @brief Add a single entry to the gallery.
     */
    void AddToGallery(const IdentityEntry& entry);

    /** @brief Get the total number of gallery entries */
    size_t GallerySize() const;

    /** @brief Clear all short-term entries */
    void ClearShortTerm();

private:
    // Short-term store: track_id → (embedding, identity)
    struct ShortTermEntry {
        std::vector<float> embedding;
        std::string identity_id;
        std::string identity_name;
    };
    std::unordered_map<int, ShortTermEntry> short_term_;
    mutable std::mutex short_term_mutex_;

    // Gallery store (FAISS-based or brute-force fallback)
    std::vector<IdentityEntry> gallery_entries_;
    mutable std::mutex gallery_mutex_;

    // FAISS index (opaque pointer to avoid header dependency)
    void* faiss_index_ = nullptr;

    void BuildFaissIndex();
    void SearchGallery(const std::vector<float>& embedding, int k,
                       std::vector<size_t>& indices,
                       std::vector<float>& distances) const;

    static float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
};

} // namespace face_rec

#endif // GPU_FACE_RECOGNITION_FACE_INDEX_H
