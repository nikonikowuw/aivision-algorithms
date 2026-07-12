#include "common/logger.h"
#include "pipeline/face_index.h"
#include "preprocess/image_utils.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool TestNV12Stride() {
    const std::vector<uint8_t> nv12 = {
        100, 100, 0, 0,
        100, 100, 0, 0,
        128, 128, 0, 0,
    };
    std::vector<uint8_t> bgr(2 * 2 * 3);
    if (!image_utils::NV12ToBGR(nv12.data(), 2, 2, 4, 4, bgr.data())) return false;
    for (uint8_t value : bgr) {
        if (std::abs(static_cast<int>(value) - 100) > 1) return false;
    }
    return !image_utils::NV12ToBGR(nv12.data(), 3, 2, 4, 4, bgr.data());
}

bool TestFaceIndex() {
    face_rec::FaceIndex index;
    index.Initialize(0.0f);

    face_rec::FaceIndex::Snapshot snapshot;
    snapshot.version = "test";
    snapshot.embedding_dim = 3;
    snapshot.identities = {
        {"alice", "Alice", {1.0f, 0.0f, 0.0f}},
        {"bob", "Bob", {0.0f, 1.0f, 0.0f}},
        {"alice", "Alice 2", {0.8f, 0.2f, 0.0f}},
    };
    snapshot.embeddings = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.8f, 0.2f, 0.0f,
    };
    index.Replace(std::move(snapshot));

    const float query[] = {1.0f, 0.0f, 0.0f};
    const auto matches = index.SearchTopK(query, 3, 2);
    return matches.size() == 2 && matches[0].identity_id == "alice" &&
           std::abs(matches[0].similarity - 1.0f) < 1e-6f &&
           matches[1].identity_id == "bob";
}

bool TestTimerReset() {
    face_rec::StageTimer timer;
    timer.Start();
    timer.Stop();
    timer.Reset();
    return timer.ElapsedMs() == 0.0;
}

}  // namespace

int main() {
    if (!TestNV12Stride()) {
        std::cerr << "NV12 stride test failed\n";
        return 1;
    }
    if (!TestFaceIndex()) {
        std::cerr << "Face index test failed\n";
        return 1;
    }
    if (!TestTimerReset()) {
        std::cerr << "Timer reset test failed\n";
        return 1;
    }
    return 0;
}
