#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// For half precision
#include <cuda_fp16.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/TypeCast.h>

// Helper methods

void caughtError(std::exception const &e) { getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, e.what()); }

void logInfo(char const *msg) { getLogger()->log(nvinfer1::ILogger::Severity::kINFO, msg); }

void logVerbose(char const *msg) { getLogger()->log(nvinfer1::ILogger::Severity::kVERBOSE, msg); }

#define PLUGIN_ASSERT(val) reportAssertion((val), #val, __FILE__, __LINE__)
void reportAssertion(bool success, char const *msg, char const *file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    std::abort();
  }
}

#define PLUGIN_VALIDATE(val) reportValidation((val), #val, __FILE__, __LINE__)
void reportValidation(bool success, char const *msg, char const *file, int32_t line) {
  if (!success) {
    std::ostringstream stream;
    stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
  }
}

at::ScalarType getATenDtype(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return at::kFloat;
  case nvinfer1::DataType::kHALF:
    return at::kHalf;
  default:
    throw std::runtime_error("Unsupported TensorRT data type");
  }
}

namespace codetr {
extern void ms_deform_attn_forward_reference(const at::Tensor &value, const at::Tensor &spatial_shapes,
                                             const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
                                             const at::Tensor &attn_weight, at::Tensor &output,
                                             const int64_t im2col_step);
}

// ---------------------------------------------------------------------------
// DeformableAttentionPlugin
// ---------------------------------------------------------------------------

// In IPluginV3 interface, the plugin name, version, and name space must be
// specified for the plugin and plugin creator exactly the same.
constexpr char const *const kDEFORM_ATTN_PLUGIN_NAME{"DeformableAttentionPlugin"};
constexpr char const *const kDEFORM_ATTN_PLUGIN_VERSION{"1"};
constexpr char const *const kDEFORM_ATTN_PLUGIN_NAMESPACE{""};

namespace nvinfer1 {
namespace plugin {

struct DeformableAttentionParameters {
  int64_t im2col_step;
};

class DeformableAttentionPlugin : public IPluginV3,
                                  public IPluginV3OneCore,
                                  public IPluginV3OneBuild,
                                  public IPluginV3OneRuntime {
public:
  // Construct from user param
  explicit DeformableAttentionPlugin(DeformableAttentionParameters const &params) : mParams{params} {
    initFieldsToSerialize();
  }

  ~DeformableAttentionPlugin() override = default;

  // IPluginV3 Methods

  IPluginCapability *getCapabilityInterface(PluginCapabilityType type) noexcept override {
    try {
      // Build capability: Refers to plugin attributes and behaviors that the
      // plugin must exhibit for the TensorRT builder.
      if (type == PluginCapabilityType::kBUILD) {
        return static_cast<IPluginV3OneBuild *>(this);
      }
      // Runtime capability: Refers to plugin attributes and behaviors that the
      // plugin must exhibit for it to be executable, either during auto-tuning
      // in the TensorRT build phase or inference in the TensorRT runtime phase
      if (type == PluginCapabilityType::kRUNTIME) {
        return static_cast<IPluginV3OneRuntime *>(this);
      }
      // Core capability: Refers to plugin attributes and behaviors common to
      // both the build and runtime phases of a plugin’s lifetime.
      PLUGIN_ASSERT(type == PluginCapabilityType::kCORE);
      return static_cast<IPluginV3OneCore *>(this);
    } catch (std::exception const &e) {
      caughtError(e);
    }
    return nullptr;
  }

  IPluginV3 *clone() noexcept override {
    // It's possible to encounter errors during cloning.
    // For example, if the memory to allocate is insufficient, exceptions can be
    // thrown.
    try {
      IPluginV3 *const plugin{new DeformableAttentionPlugin{mParams}};
      return plugin;
    } catch (std::exception const &e) {
      caughtError(e);
    }
    return nullptr;
  }

  // IPluginV3OneCore Methods

  char const *getPluginName() const noexcept override { return kDEFORM_ATTN_PLUGIN_NAME; }

  char const *getPluginVersion() const noexcept override { return kDEFORM_ATTN_PLUGIN_VERSION; }

  char const *getPluginNamespace() const noexcept override { return kDEFORM_ATTN_PLUGIN_NAMESPACE; }

  // IPluginV3OneBuild Methods

  // Number of plugin outputs
  int32_t getNbOutputs() const noexcept override { return 1; }

  int32_t configurePlugin(DynamicPluginTensorDesc const *in, int32_t nbInputs, DynamicPluginTensorDesc const *out,
                          int32_t nbOutputs) noexcept override {

    // Communicates the number of inputs and outputs, dimensions, and datatypes
    // of all inputs and outputs, broadcast information for all inputs and
    // outputs, the chosen plugin format, and maximum batch size. At this point,
    // the plugin sets up its internal state and selects the most appropriate
    // algorithm and data structures for the given configuration. Note: Resource
    // allocation is not allowed in this API because it causes a resource leak.

    // This member function will only be called during engine build time.

    PLUGIN_ASSERT(nbInputs == 5);
    PLUGIN_ASSERT(nbOutputs == 1);

    // value (bs, num_keys, num_heads, dim_per_head)
    PLUGIN_ASSERT(in[0].desc.dims.nbDims == 4);

    // spatial_shapes (num_levels, 2)
    PLUGIN_ASSERT(in[1].desc.dims.nbDims == 2);

    // level_start_index (num_levels,)
    PLUGIN_ASSERT(in[2].desc.dims.nbDims == 1);

    // sampling_loc (bs, num_queries, num_heads, num_levels, num_points, 2)
    PLUGIN_ASSERT(in[3].desc.dims.nbDims == 6);

    // attn_weight (bs, num_queries, num_heads, num_levels, num_points)
    PLUGIN_ASSERT(in[4].desc.dims.nbDims == 5);

    // output (bs, num_queries, num_heads * dim_per_head)
    PLUGIN_ASSERT(out[0].desc.dims.nbDims == 3);

    // Check bs
    auto bs = in[0].desc.dims.d[0];
    PLUGIN_ASSERT(bs == in[3].desc.dims.d[0]);
    PLUGIN_ASSERT(bs == in[4].desc.dims.d[0]);
    PLUGIN_ASSERT(bs == out[0].desc.dims.d[0]);

    // Check num_queries
    auto num_queries = in[3].desc.dims.d[1];
    PLUGIN_ASSERT(num_queries == in[4].desc.dims.d[1]);
    PLUGIN_ASSERT(num_queries == out[0].desc.dims.d[1]);

    // Check num_heads
    auto num_heads = in[0].desc.dims.d[2];
    PLUGIN_ASSERT(num_heads == in[3].desc.dims.d[2]);
    PLUGIN_ASSERT(num_heads == in[4].desc.dims.d[2]);

    // Check num_levels
    auto num_levels = in[1].desc.dims.d[0];
    PLUGIN_ASSERT(num_levels == in[2].desc.dims.d[0]);
    PLUGIN_ASSERT(num_levels == in[3].desc.dims.d[3]);
    PLUGIN_ASSERT(num_levels == in[4].desc.dims.d[3]);

    // Check num_points
    auto num_points = in[3].desc.dims.d[4];
    PLUGIN_ASSERT(num_points == in[4].desc.dims.d[4]);

    // Check output_dim
    auto dim_per_head = in[0].desc.dims.d[3];
    auto output_dim = num_heads * dim_per_head;
    PLUGIN_ASSERT(output_dim == out[0].desc.dims.d[2]);

    return 0;
  }

  bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const *inOut, int32_t nbInputs,
                                 int32_t nbOutputs) noexcept override {
    // For this method inputs are numbered 0..(nbInputs-1) and outputs are
    // numbered nbInputs..(nbInputs+nbOutputs-1). Using this numbering, pos is
    // an index into InOut, where 0 <= pos < nbInputs+nbOutputs.
    PLUGIN_ASSERT(nbInputs == 5 && nbOutputs == 1 && pos < nbInputs + nbOutputs);

    // 0: value             float
    // 1: spatial_shapes    int64
    // 2: level_start_index int64
    // 3: sampling_loc      float
    // 4: attn_weight       float
    // 5: output           float

    const auto first_type = inOut[0].desc.type;

    const auto type = inOut[pos].desc.type;
    const auto format = inOut[pos].desc.format;

    bool isValidCombination = format == TensorFormat::kLINEAR;

    if (pos == 0 || pos == 3 || pos == 4 || pos == 5) {
      isValidCombination &= (type == DataType::kFLOAT || type == DataType::kHALF);
      isValidCombination &= type == first_type;
    } else if (pos == 1 || pos == 2) {
      isValidCombination &= type == DataType::kINT64;
    }
    return isValidCombination;
  }

  int32_t getOutputDataTypes(DataType *outputTypes, int32_t nbOutputs, DataType const *inputTypes,
                             int32_t nbInputs) const noexcept override {
    PLUGIN_ASSERT(nbInputs == 5);
    PLUGIN_ASSERT(nbOutputs == 1);
    // The output type is the same as the input type.
    outputTypes[0] = inputTypes[0];
    return 0;
  }

  int32_t getOutputShapes(DimsExprs const *inputs, int32_t nbInputs, DimsExprs const *shapeInputs,
                          int32_t nbShapeInputs, DimsExprs *outputs, int32_t nbOutputs,
                          IExprBuilder &exprBuilder) noexcept override {
    //   inputs[0] -> value (bs, num_keys, num_heads, dim_per_head)
    //   inputs[3] -> sampling_loc (bs, num_queries, num_heads, num_levels,
    //                              num_points, 2)
    //   outputs[0] -> output (bs, num_queries, num_heads * dim_per_head)

    PLUGIN_ASSERT(nbInputs == 5);
    PLUGIN_ASSERT(nbOutputs == 1);
    PLUGIN_ASSERT(inputs != nullptr);
    PLUGIN_ASSERT(inputs[0].nbDims == 4);
    PLUGIN_ASSERT(inputs[3].nbDims == 6);

    auto bs = inputs[0].d[0];
    auto num_heads = inputs[0].d[2];
    auto dim_per_head = inputs[0].d[3];
    auto num_queries = inputs[3].d[1];

    outputs[0].nbDims = 3;
    outputs[0].d[0] = bs;
    outputs[0].d[1] = num_queries;
    outputs[0].d[2] = exprBuilder.operation(DimensionOperation::kPROD, *num_heads, *dim_per_head);
    return 0;
  }

  // IPluginV3OneRuntime Methods

  int32_t enqueue(PluginTensorDesc const *inputDesc, PluginTensorDesc const *outputDesc, void const *const *inputs,
                  void *const *outputs, void *workspace, cudaStream_t stream) noexcept override {
    // void const* const* inputs: A pointer to a constant pointer to constant
    // untyped data
    //   - array of input pointers where each element is a pointer to some input
    //   data
    //   - Neither the array elements (the pointers) nor the data they point to
    //   can be modified through this parameter
    //   - he parameter itself can be redirected to a different array

    PLUGIN_ASSERT(inputs[0] != nullptr);
    PLUGIN_ASSERT(inputs[1] != nullptr);
    PLUGIN_ASSERT(inputs[2] != nullptr);
    PLUGIN_ASSERT(inputs[3] != nullptr);
    PLUGIN_ASSERT(inputs[4] != nullptr);
    PLUGIN_ASSERT(outputs[0] != nullptr);

    // We parse shapes from the stored dims
    // [0] -> (bs, num_keys, num_heads, dim_per_head)
    // [3] -> (bs, num_queries, num_heads, num_levels, num_points, 2)
    const auto &value_dims = inputDesc[0].dims;
    const auto &sampling_loc_dims = inputDesc[3].dims;

    int bs = value_dims.d[0];
    int num_keys = value_dims.d[1];
    int num_heads = value_dims.d[2];
    int dim_per_head = value_dims.d[3];

    int num_queries = sampling_loc_dims.d[1];
    int num_levels = sampling_loc_dims.d[3];
    int num_points = sampling_loc_dims.d[4];

    // Helper function to map DataType to string
    auto dataTypeToString = [](nvinfer1::DataType type) -> std::string {
      switch (type) {
      case nvinfer1::DataType::kFLOAT:
        return "float32";
      case nvinfer1::DataType::kHALF:
        return "float16";
      case nvinfer1::DataType::kINT32:
        return "int32";
      case nvinfer1::DataType::kINT64:
        return "int64";
      default:
        return "unknown";
      }
    };
    // Log tensor shapes for debugging
    std::stringstream ss;
    ss << "DeformableAttentionPlugin::enqueue: " << std::endl
       << "  value shape: (" << bs << ", " << num_keys << ", " << num_heads << ", " << dim_per_head << ")" << std::endl
       << "  sampling_loc shape: (" << bs << ", " << num_queries << ", " << num_heads << ", " << num_levels << ", "
       << num_points << ", 2)" << std::endl
       << "  attn_weight shape: (" << bs << ", " << num_queries << ", " << num_heads << ", " << num_levels << ", "
       << num_points << ")" << std::endl
       << "  output shape: (" << bs << ", " << num_queries << ", " << num_heads * dim_per_head << ")" << std::endl
       << "  nullptr checks: " << std::endl
       << "  inputs[0]: " << (inputs[0] == nullptr ? "true" : "false")
       << ", dtype: " << dataTypeToString(inputDesc[0].type) << std::endl
       << "  inputs[1]: " << (inputs[1] == nullptr ? "true" : "false")
       << " dtype: " << dataTypeToString(inputDesc[1].type) << std::endl
       << "  inputs[2]: " << (inputs[2] == nullptr ? "true" : "false")
       << " dtype: " << dataTypeToString(inputDesc[2].type) << std::endl
       << "  inputs[3]: " << (inputs[3] == nullptr ? "true" : "false")
       << " dtype: " << dataTypeToString(inputDesc[3].type) << std::endl
       << "  inputs[4]: " << (inputs[4] == nullptr ? "true" : "false")
       << " dtype: " << dataTypeToString(inputDesc[4].type) << std::endl
       << "  outputs[0]: " << (outputs[0] == nullptr ? "true" : "false")
       << " dtype: " << dataTypeToString(outputDesc[0].type) << std::endl;
    logVerbose(ss.str().c_str());

    DataType dtype = inputDesc[0].type;

    at::ScalarType scalar_type;
    try {
      scalar_type = getATenDtype(dtype);
    } catch (const std::runtime_error &e) {
      caughtError(e);
      return 1;
    }
    auto options = at::TensorOptions().dtype(scalar_type).device(at::kCUDA);
    auto options_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);

    // Wrap the provided cudaStream_t in a c10::cuda::CUDAStream
    c10::cuda::CUDAStream cuda_stream = c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());

    // Use CUDAStreamGuard to set this stream as current
    c10::cuda::CUDAStreamGuard stream_guard(cuda_stream);

    // We're telling PyTorch to use externally-managed memory provied by
    // TensorRT
    at::Tensor value = at::from_blob(const_cast<void *>(inputs[0]), {bs, num_keys, num_heads, dim_per_head}, options);
    // reinterpret_cast<int64_t*>
    at::Tensor spatial_shapes = at::from_blob(const_cast<void *>(inputs[1]), {num_levels, 2}, options_long);
    // reinterpret_cast<int64_t*>
    at::Tensor level_start_index = at::from_blob(const_cast<void *>(inputs[2]), {num_levels}, options_long);
    at::Tensor sampling_loc =
        at::from_blob(const_cast<void *>(inputs[3]), {bs, num_queries, num_heads, num_levels, num_points, 2}, options);

    at::Tensor attn_weight =
        at::from_blob(const_cast<void *>(inputs[4]), {bs, num_queries, num_heads, num_levels, num_points}, options);

    // Wrap outputs[0] as the output tensor
    at::Tensor trt_output =
        at::from_blob(const_cast<void *>(outputs[0]), {bs, num_queries, num_heads * dim_per_head}, options);
    codetr::ms_deform_attn_forward_reference(value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
                                             trt_output, mParams.im2col_step);

    return 0;
  }

  // Called during both the build-phase and runtime phase before enqueue() to
  // communicate the input and output shapes for the subsequent enqueue(). The
  // output PluginTensorDesc will contain wildcards (-1) for any data-dependent
  // dimensions specified through getOutputShapes().
  int32_t onShapeChange(PluginTensorDesc const *in, int32_t nbInputs, PluginTensorDesc const *out,
                        int32_t nbOutputs) noexcept override {

    return 0;
  }

  IPluginV3 *attachToContext(IPluginResourceContext *context) noexcept override { return clone(); }

  PluginFieldCollection const *getFieldsToSerialize() noexcept override { return &mFCToSerialize; }

  size_t getWorkspaceSize(DynamicPluginTensorDesc const *inputs, int32_t nbInputs,
                          DynamicPluginTensorDesc const *outputs, int32_t nbOutputs) const noexcept override {
    return 0;
  }

private:
  DeformableAttentionParameters mParams;
  std::vector<nvinfer1::PluginField> mDataToSerialize;
  nvinfer1::PluginFieldCollection mFCToSerialize;

  void initFieldsToSerialize() {
    // Serialize DeformableAttentionParameters
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back(nvinfer1::PluginField("parameters", &mParams, PluginFieldType::kUNKNOWN,
                                                        sizeof(DeformableAttentionParameters)));
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
  }
};

// ---------------------------------------------------------------------------
// DeformableAttentionPluginCreator
// ---------------------------------------------------------------------------
class DeformableAttentionPluginCreator : public IPluginCreatorV3One {
public:
  DeformableAttentionPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("im2col_step", nullptr, PluginFieldType::kINT64, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  ~DeformableAttentionPluginCreator() override = default;

  nvinfer1::PluginFieldCollection const *getFieldNames() noexcept override {
    // This is only used in the build phase.
    return &mFC;
  }

  IPluginV3 *createPlugin(char const *name, PluginFieldCollection const *fc, TensorRTPhase phase) noexcept override {
    // The build phase and the deserialization phase are handled differently.
    if (phase == TensorRTPhase::kBUILD) {
      try {
        int64_t im2col_step = 64; // default
        for (int i = 0; i < fc->nbFields; ++i) {
          const auto &f = fc->fields[i];
          if (strcmp(f.name, "im2col_step") == 0 && f.type == PluginFieldType::kINT64) {
            im2col_step = *static_cast<const int64_t *>(f.data);
          }
        }
        DeformableAttentionParameters const params{im2col_step};
        DeformableAttentionPlugin *const plugin{new DeformableAttentionPlugin{params}};
        return plugin;
      } catch (std::exception const &e) {
        caughtError(e);
      }
      return nullptr;
    } else if (phase == TensorRTPhase::kRUNTIME) {
      // The attributes from the serialized plugin will be passed via fc.
      try {

        nvinfer1::PluginField const *fields{fc->fields};
        int32_t nbFields{fc->nbFields};
        PLUGIN_VALIDATE(nbFields == 1);

        char const *attrName = fields[0].name;
        PLUGIN_VALIDATE(!strcmp(attrName, "parameters"));
        PLUGIN_VALIDATE(fields[0].type == nvinfer1::PluginFieldType::kUNKNOWN);
        PLUGIN_VALIDATE(fields[0].length == sizeof(DeformableAttentionParameters));
        DeformableAttentionParameters params{*(static_cast<DeformableAttentionParameters const *>(fields[0].data))};

        DeformableAttentionPlugin *const plugin{new DeformableAttentionPlugin{params}};
        return plugin;
      } catch (std::exception const &e) {
        caughtError(e);
      }
      return nullptr;
    } else {
      return nullptr;
    }
    return nullptr;
  }

  char const *getPluginNamespace() const noexcept override { return kDEFORM_ATTN_PLUGIN_NAMESPACE; }

  char const *getPluginName() const noexcept override { return kDEFORM_ATTN_PLUGIN_NAME; }

  char const *getPluginVersion() const noexcept override { return kDEFORM_ATTN_PLUGIN_VERSION; }

private:
  nvinfer1::PluginFieldCollection mFC;
  std::vector<nvinfer1::PluginField> mPluginAttributes;
};

// Register the plugin with TensorRT's global registry so it can be discovered
REGISTER_TENSORRT_PLUGIN(DeformableAttentionPluginCreator);

} // namespace plugin
} // namespace nvinfer1

class ThreadSafeLoggerFinder {
public:
  ThreadSafeLoggerFinder() = default;

  // Set the logger finder.
  void setLoggerFinder(nvinfer1::ILoggerFinder *finder) {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder == nullptr && finder != nullptr) {
      mLoggerFinder = finder;
    }
  }

  // Get the logger.
  nvinfer1::ILogger *getLogger() noexcept {
    std::lock_guard<std::mutex> lk(mMutex);
    if (mLoggerFinder != nullptr) {
      return mLoggerFinder->findLogger();
    }
    return nullptr;
  }

private:
  nvinfer1::ILoggerFinder *mLoggerFinder{nullptr};
  std::mutex mMutex;
};

ThreadSafeLoggerFinder gLoggerFinder;

// Not exposing this function to the user to get the plugin logger for the
// moment. Can switch the plugin logger to this in the future.

// ILogger* getPluginLogger()
// {
//     return gLoggerFinder.getLogger();
// }

extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder *finder) { gLoggerFinder.setLoggerFinder(finder); }

extern "C" nvinfer1::IPluginCreatorInterface *const *getPluginCreators(int32_t &nbCreators) {
  nbCreators = 1;
  static nvinfer1::plugin::DeformableAttentionPluginCreator creator{};
  static nvinfer1::IPluginCreatorInterface *const pluginCreatorList[] = {&creator};
  return pluginCreatorList;
}