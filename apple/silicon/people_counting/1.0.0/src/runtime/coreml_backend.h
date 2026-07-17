/**
 * @file coreml_backend.h
 * @brief CoreML native inference backend
 */

#ifndef PEOPLE_COUNTING_COREML_BACKEND_H
#define PEOPLE_COUNTING_COREML_BACKEND_H

#include <string>
#include <vector>
#include <memory>

namespace people_count {

struct ModelInput {
    const void* data;           // Zero-copy pointer to flat float data
    size_t size;                // Element count
    const int64_t* shape;       // Shape dimensions
    size_t shape_len;           // Shape length
};

struct ModelOutput {
    std::string name;
    std::vector<float> buffer;
    int64_t shape[4];
};

class CoreMLBackend {
public:
    CoreMLBackend();
    ~CoreMLBackend();

    bool Load(const std::string& model_path);
    bool Run(const std::vector<ModelInput>& inputs,
             std::vector<ModelOutput>* outputs);
    void Unload();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace people_count

#endif // PEOPLE_COUNTING_COREML_BACKEND_H
