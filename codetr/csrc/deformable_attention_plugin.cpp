#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimePlugin.h>
#include <NvInferRuntime.h>

#include <cuda_runtime.h>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

// For half precision
#include <cuda_fp16.h>

// Helper methods

void caughtError(std::exception const& e)
{
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what());
}

void logInfo(char const* msg)
{
    getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg);
}

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
void reportAssertion(bool success, char const* msg, char const* file,
                     int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Assertion failed: " << msg << std::endl
               << file << ':' << line << std::endl
               << "Aborting..." << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                         stream.str().c_str());
        std::abort();
    }
}

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
void reportValidation(bool success, char const* msg, char const* file,
                      int32_t line)
{
    if (!success)
    {
        std::ostringstream stream;
        stream << "Validation failed: " << msg << std::endl
               << file << ':' << line << std::endl;
        getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR,
                         stream.str().c_str());
    }
}


// If your custom CUDA code is in a separate .cu file, forward declare it here:
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

// ---------------------------------------------------------------------------
// DeformableAttentionPluginV3
// ---------------------------------------------------------------------------

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const* const kDEFORM_ATTN_PLUGIN_NAME{"DeformableAttentionPluginV3"};
constexpr char const* const kDEFORM_ATTN_PLUGIN_VERSION{"1"};
constexpr char const* const kDEFORM_ATTN_PLUGIN_NAMESPACE{""};

namespace nvinfer1 {

class DeformableAttentionPluginV3 : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    // Construct from user param
    explicit DeformableAttentionPluginV3(int im2colStep)
        : mIm2colStep(im2colStep)
        , mDataType(DataType::kFLOAT)
        , mValueDims{}
        , mSamplingLocDims{}
        , mNamespace("")
    {
    }

    // Construct from serialization
    DeformableAttentionPluginV3(const void* serialData, size_t serialLength)
        : mDataType(DataType::kFLOAT)
        , mValueDims{}
        , mSamplingLocDims{}
        , mNamespace("")
    {
        const int* p = reinterpret_cast<const int*>(serialData);
        mIm2colStep = *p;
    }

    ~DeformableAttentionPluginV3() override = default;

    // IPluginV3 Methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        try
        {
            if (type == PluginCapabilityType::kBUILD)
            {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == PluginCapabilityType::kRUNTIME)
            {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }
    
    IPluginV3* clone() noexcept override
    {
        // It's possible to encounter errors during cloning.
        // For example, if the memory to allocate is insufficient, exceptions can be
        // thrown.
        try
        {
            IPluginV3* const plugin{new DeformableAttentionPluginV3{mIm2colStep}};
            return plugin;
        }
        catch (std::exception const& e)
        {
            caughtError(e);
        }
        return nullptr;
    }

    // IPluginV3OneCore Methods

    char const* getPluginName() const noexcept override
    {
        return kDEFORM_ATTN_PLUGIN_NAME;
    }
    
    char const* getPluginVersion() const noexcept override
    {
        return kDEFORM_ATTN_PLUGIN_VERSION;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return kDEFORM_ATTN_PLUGIN_NAMESPACE;
    }

    // IPluginV3OneBuild Methods

    // Number of plugin outputs
    int32_t getNbOutputs() const noexcept override
    {
        return 1;
    }

    int32_t configurePlugin(
        DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        PLUGIN_ASSERT(nbInputs == 5);
        PLUGIN_ASSERT(nbOutputs == 1);
        PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);
        PLUGIN_ASSERT(in[1].desc.dims.nbDims == 2);
        PLUGIN_ASSERT(in[2].desc.dims.nbDims == 2);
        PLUGIN_ASSERT(in[3].desc.dims.nbDims == 6);
        PLUGIN_ASSERT(in[4].desc.dims.nbDims == 4);
        PLUGIN_ASSERT(in[0].desc.dims.d[0] == in[3].desc.dims.d[0]);
        PLUGIN_ASSERT(in[0].desc.dims.d[2] == in[4].desc.dims.d[1]);
        PLUGIN_ASSERT(in[0].desc.dims.d[3] == in[4].desc.dims.d[2]);
        PLUGIN_ASSERT(in[0].desc.dims.d[1] == in[1].desc.dims.d[0]);
        PLUGIN_ASSERT(in[0].desc.dims.d[2] == in[3].desc.dims.d[2]);
        PLUGIN_ASSERT(in[0].desc.dims.d[3] == in[4].desc.dims.d[3]);
        PLUGIN_ASSERT(in[1].desc.dims.d[0] == in[2].desc.dims.d[0]);
        PLUGIN_ASSERT(in[1].desc.dims.d[0] == in[3].desc.dims.d[3]);
        PLUGIN_ASSERT(in[1].desc.dims.d[0] == in[4].desc.dims.d[0]);
        PLUGIN_ASSERT(in[2].desc.dims.d[0] == in[3].desc.dims.d[4]);
        PLUGIN_ASSERT(in[2].desc.dims.d[0] == in[4].desc.dims.d[1]);
        PLUGIN_ASSERT(in[3].desc.dims.d[0] == in[4].desc.dims.d[0]);
        PLUGIN_ASSERT(in[3].desc.dims.d[1] == in[4].desc.dims.d[1]);
        PLUGIN_ASSERT(in[3].desc.dims.d[2] == in[4].desc.dims.d[2]);
        PLUGIN_ASSERT(in[3].desc.dims.d[3] == in[4].desc.dims.d[3]);
        PLUGIN_ASSERT(in[3].desc.dims.d[4] == in[4].desc.dims.d[4]);
        PLUGIN_ASSERT(in[3].desc.dims.d[5] == 2);
        PLUGIN_ASSERT(in[4].desc.dims.d[5] == 2);
        PLUGIN_ASSERT(in[0].desc.type == in[1].desc.type);
        PLUGIN_ASSERT(in[0].desc.type == in[2].desc.type);
        PLUGIN_ASSERT(in[0].desc.type == in[3].desc.type);
        PLUGIN_ASSERT(in[0].desc.type == in[4].desc.type);
        PLUGIN_ASSERT(in[0].desc.format == in[1].desc.format);
        PLUGIN_ASSERT(in[0].desc.format == in[2].desc.format);
        PLUGIN_ASSERT(in[0].desc.format == in[3].desc.format);
        PLUGIN_ASSERT(in[0].desc.format == in[4].desc.format);
        PLUGIN_ASSERT(in[0].desc.type == out[0].desc.type);
        PLUGIN_ASSERT(in[0].desc.format == out[0].desc.format);

        // TODO FIX THIS

        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos,
        DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override
    {
        // For this method inputs are numbered 0..(nbInputs-1) and outputs are
        // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
        // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
        PLUGIN_ASSERT(nbInputs == 5 && nbOutputs == 1 &&
            pos < nbInputs + nbOutputs);

        bool isValidCombination = false;


        // Just check the type of the first input tensor
        const auto type = inOut[0].desc.type;
        const auto format = inOut[0].desc.format;
        isValidCombination = (type == DataType::kFLOAT || type == DataType::kHALF) && format == TensorFormat::kLINEAR;
        
        // TODO check the types of the other inputs and outputs

        // Make sure the input tensor and output tensor types and formats are same.
        isValidCombination &=
            (pos < nbInputs || (inOut[pos].desc.format == inOut[0].desc.format &&
                                inOut[pos].desc.type == inOut[0].desc.type));

        return isValidCombination;
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes,
        int32_t nbOutputs,
        DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        PLUGIN_ASSERT(nbInputs == 5);
        PLUGIN_ASSERT(nbOutputs == 1);
        // The output type is the same as the input type.
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(
        DimsExprs const* inputs,
        int32_t nbInputs,
        DimsExprs const* shapeInputs,
        int32_t nbShapeInputs,
        DimsExprs* outputs,
        int32_t nbOutputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        // We assume:
        //   inputs[0] -> value (bs, num_keys, num_heads, dim_per_head)
        //   inputs[3] -> sampling_loc (bs, num_queries, num_heads, num_levels, num_points, 2)
        const int bs        = inputs[0].d[0];
        const nvinfer1::IDimensionExpr* num_heads_expr = inputs[0].d[2];
        const int dim_per_head = inputs[0].d[3];
        const int num_queries  = inputs[3].d[1];


        // TODO FINISH THIS
        auto bs = inputs[0].d[0];
        auto num_queries = inputs[3].d[1];
        auto embed_dims = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);

        DimsExprs out;
        out.nbDims = 3;
        out.d[0] = bs;
        out.d[1] = num_queries;
        out.d[2] = embed_dims;
        return out;
    }


    // For static shape usage in IPluginV3
    Dims getOutputDimensions(int32_t /*index*/, const Dims* inputs, int32_t /*nbInputs*/) noexcept override
    {
        // We assume:
        //   inputs[0] -> value (bs, num_keys, num_heads, dim_per_head)
        //   inputs[3] -> sampling_loc (bs, num_queries, num_heads, num_levels, num_points, 2)
        const int bs        = inputs[0].d[0];
        const int num_heads = inputs[0].d[2];
        const int dim_per_head = inputs[0].d[3];
        const int num_queries  = inputs[3].d[1];

        Dims out;
        out.nbDims = 3;
        out.d[0] = bs;
        out.d[1] = num_queries;
        out.d[2] = num_heads * dim_per_head; // embed_dims
        return out;
    }

    // Which type + format do we accept
    bool supportsFormat(DataType type, TensorFormat format) const noexcept override
    {
        if (format != TensorFormat::kLINEAR)
            return false;
        if (type != DataType::kFLOAT && type != DataType::kHALF)
            return false;
        return true;
    }

    // Called once at engine build time
    void configureWithFormat(
        const Dims* inputDims, int32_t nbInputs,
        const Dims* outputDims, int32_t nbOutputs,
        DataType type, DataType /*outputType*/,
        TensorFormat inFormat, TensorFormat /*outFormat*/,
        int32_t /*maxBatchSize*/) noexcept override
    {
        mDataType = type;

        // We'll store only the dims we need
        // Typically: 0 = value, 3 = sampling_loc
        if (nbInputs >= 4)
        {
            mValueDims       = inputDims[0];
            mSamplingLocDims = inputDims[3];
        }
    }

    // Called once at engine init
    int32_t initialize() noexcept override
    {
        return 0; // no-op
    }

    // Called once at engine tear-down
    void terminate() noexcept override
    {
        // no-op
    }

    // If we needed scratch memory, we specify it here
    size_t getWorkspaceSize(int32_t /*maxBatchSize*/) const noexcept override
    {
        return 0;
    }

    // The main inference entry point
    int32_t enqueue(
        int32_t /*batchSize*/,
        const void* const* inputs,
        void* const* outputs,
        void* /*workspace*/,
        cudaStream_t stream) noexcept override
    {
        // We parse shapes from the stored dims
        // [0] -> (bs, num_keys, num_heads, dim_per_head)
        int bs           = mValueDims.d[0];
        int num_keys     = mValueDims.d[1];
        int num_heads    = mValueDims.d[2];
        int dim_per_head = mValueDims.d[3];

        // [3] -> (bs, num_queries, num_heads, num_levels, num_points, 2)
        int num_queries  = mSamplingLocDims.d[1];
        int num_levels   = mSamplingLocDims.d[3];
        int num_points   = mSamplingLocDims.d[4];

        if (mDataType == DataType::kFLOAT)
        {
            const float* value_ptr             = static_cast<const float*>(inputs[0]);
            const int64_t* spatial_shapes_ptr  = static_cast<const int64_t*>(inputs[1]);
            const int64_t* level_start_index_ptr = static_cast<const int64_t*>(inputs[2]);
            const float* sampling_loc_ptr      = static_cast<const float*>(inputs[3]);
            const float* attn_weight_ptr       = static_cast<const float*>(inputs[4]);
            float* output_ptr                  = static_cast<float*>(outputs[0]);

            multi_scale_deformable_attention_cuda_forward<float>(
                value_ptr,
                spatial_shapes_ptr,
                level_start_index_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                output_ptr,
                bs,
                num_keys,
                num_queries,
                num_heads,
                dim_per_head,
                num_levels,
                num_points,
                mIm2colStep,
                stream
            );
        }
        else if (mDataType == DataType::kHALF)
        {
            const half* value_ptr            = static_cast<const half*>(inputs[0]);
            const int64_t* spatial_shapes_ptr  = static_cast<const int64_t*>(inputs[1]);
            const int64_t* level_start_index_ptr = static_cast<const int64_t*>(inputs[2]);
            const half* sampling_loc_ptr     = static_cast<const half*>(inputs[3]);
            const half* attn_weight_ptr      = static_cast<const half*>(inputs[4]);
            half* output_ptr                 = static_cast<half*>(outputs[0]);

            multi_scale_deformable_attention_cuda_forward<half>(
                value_ptr,
                spatial_shapes_ptr,
                level_start_index_ptr,
                sampling_loc_ptr,
                attn_weight_ptr,
                output_ptr,
                bs,
                num_keys,
                num_queries,
                num_heads,
                dim_per_head,
                num_levels,
                num_points,
                mIm2colStep,
                stream
            );
        }
        else
        {
            std::cerr << "[DeformableAttentionPluginV3] Unsupported dtype.\n";
            return 1;
        }

        return 0;
    }

    // Plugin metadata
    const char* getPluginType() const noexcept override
    {
        // Must match the name used in the plugin creator
        return "DeformableAttentionPluginV3";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    void destroy() noexcept override
    {
        delete this;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

    // For serialization
    size_t getSerializationSize() const noexcept override
    {
        // only storing int
        return sizeof(int);
    }

    void serialize(void* buffer) const noexcept override
    {
        *reinterpret_cast<int*>(buffer) = mIm2colStep;
    }

private:
    int mIm2colStep;
    DataType mDataType;
    Dims mValueDims;
    Dims mSamplingLocDims;
    std::string mNamespace;
};


// ---------------------------------------------------------------------------
// DeformableAttentionPluginV3Creator
// ---------------------------------------------------------------------------
class DeformableAttentionPluginV3Creator : public IPluginCreator
{
public:
    DeformableAttentionPluginV3Creator()
    {
        mPluginAttributes.emplace_back(
            PluginField{"im2col_step", nullptr, PluginFieldType::kINT32, 1}
        );
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields   = mPluginAttributes.data();
        mNamespace   = "";
    }

    ~DeformableAttentionPluginV3Creator() override = default;

    const char* getPluginName() const noexcept override
    {
        // Must match plugin getPluginType()
        return "DeformableAttentionPluginV3";
    }

    const char* getPluginVersion() const noexcept override
    {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &mFC;
    }

    // createPlugin() is called when user code calls create
    IPluginV3* createPlugin(const char* /*name*/, const PluginFieldCollection* fc) noexcept override
    {
        int im2col_step = 64; // default
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const auto& f = fc->fields[i];
            if (strcmp(f.name, "im2col_step") == 0 && f.type == PluginFieldType::kINT32)
            {
                im2col_step = *static_cast<const int*>(f.data);
            }
        }
        return new DeformableAttentionPluginV3(im2col_step);
    }

    // deserializePlugin() is called when TRT loads an engine
    IPluginV3* deserializePlugin(const char* /*name*/, const void* serialData, size_t serialLength) noexcept override
    {
        return new DeformableAttentionPluginV3(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override
    {
        mNamespace = pluginNamespace ? pluginNamespace : "";
    }

    const char* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
    std::vector<PluginField> mPluginAttributes;
};

// Register the plugin with TensorRT's global registry so it can be discovered
REGISTER_TENSORRT_PLUGIN(DeformableAttentionPluginV3Creator);

} // namespace nvinfer1
