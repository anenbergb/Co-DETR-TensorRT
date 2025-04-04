#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cstring>
#include <vector>
#include <iostream>
#include <memory>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

using namespace nvinfer1;

namespace codetr {

// Forward declaration for wrapper
template <typename scalar_t>
void multi_scale_deformable_attention_cuda_forward(
    const scalar_t* value_ptr,
    const int64_t* spatial_shapes_ptr,
    const int64_t* level_start_index_ptr,
    const scalar_t* sampling_loc_ptr,
    const scalar_t* attn_weight_ptr,
    scalar_t* output_ptr,
    int bs,
    int num_keys,
    int num_queries,
    int num_heads,
    int dim_per_head,
    int num_levels,
    int num_points,
    int im2col_step,
    cudaStream_t stream);

class DeformableAttentionPlugin : public IPluginV2DynamicExt {
public:
    DeformableAttentionPlugin(int im2colStep) : im2col_step(im2colStep) {}

    DeformableAttentionPlugin(const void* data, size_t length) {
        const int* d = reinterpret_cast<const int*>(data);
        im2col_step = *d;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        return new DeformableAttentionPlugin(im2col_step);
    }

    int getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(
        int outputIndex,
        const DimsExprs* inputs,
        int nbInputs,
        IExprBuilder& exprBuilder) noexcept override {

        auto bs = inputs[0].d[0];
        auto num_queries = inputs[3].d[1];
        auto embed_dims = exprBuilder.operation(DimensionOperation::kPROD, {inputs[0].d[2], inputs[0].d[3]});

        DimsExprs out;
        out.nbDims = 3;
        out.d[0] = bs;
        out.d[1] = num_queries;
        out.d[2] = embed_dims;
        return out;
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        const auto type = inOut[0].type;
        const auto format = inOut[0].format;

        for (int i = 0; i < nbInputs + nbOutputs; ++i) {
            if (inOut[i].type != type || inOut[i].format != format)
                return false;
        }

        return (type == DataType::kFLOAT || type == DataType::kHALF) && format == TensorFormat::kLINEAR;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0;
    }

    int enqueue(const PluginTensorDesc* inputDesc,
                const PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override {

        const auto& value_dims = inputDesc[0].dims;
        const auto& sampling_loc_dims = inputDesc[3].dims;

        int bs = value_dims.d[0];
        int num_keys = value_dims.d[1];
        int num_heads = value_dims.d[2];
        int dim_per_head = value_dims.d[3];

        int num_queries = sampling_loc_dims.d[1];
        int num_levels = sampling_loc_dims.d[3];
        int num_points = sampling_loc_dims.d[4];

        DataType dtype = inputDesc[0].type;

        if (dtype == DataType::kFLOAT) {
            const float* value_ptr = static_cast<const float*>(inputs[0]);
            const int64_t* spatial_shapes_ptr = static_cast<const int64_t*>(inputs[1]);
            const int64_t* level_start_index_ptr = static_cast<const int64_t*>(inputs[2]);
            const float* sampling_loc_ptr = static_cast<const float*>(inputs[3]);
            const float* attn_weight_ptr = static_cast<const float*>(inputs[4]);
            float* output_ptr = static_cast<float*>(outputs[0]);

            multi_scale_deformable_attention_cuda_forward<float>(
                value_ptr, spatial_shapes_ptr, level_start_index_ptr,
                sampling_loc_ptr, attn_weight_ptr, output_ptr,
                bs, num_keys, num_queries, num_heads, dim_per_head,
                num_levels, num_points, im2col_step, stream);

        } else if (dtype == DataType::kHALF) {
            const __half* value_ptr = static_cast<const __half*>(inputs[0]);
            const int64_t* spatial_shapes_ptr = static_cast<const int64_t*>(inputs[1]);
            const int64_t* level_start_index_ptr = static_cast<const int64_t*>(inputs[2]);
            const __half* sampling_loc_ptr = static_cast<const __half*>(inputs[3]);
            const __half* attn_weight_ptr = static_cast<const __half*>(inputs[4]);
            __half* output_ptr = static_cast<__half*>(outputs[0]);

            multi_scale_deformable_attention_cuda_forward<__half>(
                value_ptr, spatial_shapes_ptr, level_start_index_ptr,
                sampling_loc_ptr, attn_weight_ptr, output_ptr,
                bs, num_keys, num_queries, num_heads, dim_per_head,
                num_levels, num_points, im2col_step, stream);

        } else {
            std::cerr << "Unsupported data type in DeformableAttentionPlugin." << std::endl;
            return 1;
        }

        return 0;
    }

    size_t getSerializationSize() const noexcept override { return sizeof(int); }

    void serialize(void* buffer) const noexcept override {
        *reinterpret_cast<int*>(buffer) = im2col_step;
    }

    void destroy() noexcept override { delete this; }

    const char* getPluginNamespace() const noexcept override { return ""; }
    const char* getPluginType() const noexcept override { return "DeformableAttentionPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {}

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override {
        return inputTypes[0];
    }

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept override {
        return false;
    }

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override {
        return false;
    }

private:
    int im2col_step;
};

class DeformableAttentionPluginCreator : public IPluginCreator {
public:
    DeformableAttentionPluginCreator() {
        mPluginAttributes.emplace_back(PluginField{"im2col_step", nullptr, PluginFieldType::kINT32, 1});
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* getPluginName() const noexcept override { return "DeformableAttentionPlugin"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        int im2col_step = 64;
        for (int i = 0; i < fc->nbFields; ++i) {
            if (strcmp(fc->fields[i].name, "im2col_step") == 0) {
                im2col_step = *(static_cast<const int*>(fc->fields[i].data));
            }
        }
        return new DeformableAttentionPlugin(im2col_step);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new DeformableAttentionPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override {}
    const char* getPluginNamespace() const noexcept override { return ""; }

private:
    PluginFieldCollection mFC{};
    std::vector<PluginField> mPluginAttributes;
};

REGISTER_TENSORRT_PLUGIN(DeformableAttentionPluginCreator);

} // namespace codetr