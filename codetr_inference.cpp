#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntime.h> // for createInferRuntime
#include <argparse/argparse.hpp>
#include <cassert>
#include <chrono>
#include <cuda_runtime_api.h> // for cudaMalloc/cudaMemcpy
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <torchvision/ops/nms.h>
#include <vector>

const std::vector<std::string> coco_class_names = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic_light", "fire_hydrant",  "stop_sign",    "parking_meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports_ball",  "kite",          "baseball_bat",
    "baseball_glove", "skateboard", "surfboard",     "tennis_racket", "bottle",       "wine_glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot_dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted_plant",  "bed",           "dining_table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell_phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy_bear",     "hair_drier", "toothbrush"};

std::tuple<torch::Tensor, torch::Tensor, float> preprocess_image(const cv::Mat &image, int target_height,
                                                                 int target_width) {
  std::cout << "Preprocessing image..." << std::endl;
  // Convert BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

  // Resize while maintaining aspect ratio
  int orig_height = rgb_image.rows;
  int orig_width = rgb_image.cols;
  float scale =
      std::min(static_cast<float>(target_width) / orig_width, static_cast<float>(target_height) / orig_height);

  int new_width = static_cast<int>(orig_width * scale);
  int new_height = static_cast<int>(orig_height * scale);

  cv::Mat resized;
  cv::resize(rgb_image, resized, cv::Size(new_width, new_height));

  // Create padded image
  cv::Mat padded = cv::Mat::zeros(target_height, target_width, CV_8UC3);
  resized.copyTo(padded(cv::Rect(0, 0, new_width, new_height)));

  // Convert to float32 and normalize
  cv::Mat float_image;
  padded.convertTo(float_image, CV_32F);

  // Create image mask (0 for valid pixels, 1 for padding)
  cv::Mat mask = cv::Mat::ones(target_height, target_width, CV_32F);
  mask(cv::Rect(0, 0, new_width, new_height)) = 0;

  // Convert to torch tensors
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
  torch::Tensor image_tensor = torch::from_blob(float_image.data, {target_height, target_width, 3}, options).clone();
  image_tensor = image_tensor.permute({2, 0, 1}); // HWC -> CHW
  image_tensor = image_tensor.unsqueeze(0);       // Add batch dimension

  torch::Tensor mask_tensor = torch::from_blob(mask.data, {1, target_height, target_width}, options).clone();

  // Define normalization parameters
  std::vector<float> mean = {123.675, 116.28, 103.53};
  std::vector<float> std = {58.395, 57.12, 57.375};
  // Create mean and std tensors with proper shape for broadcasting
  auto mean_tensor = torch::tensor(mean, options).view({1, 3, 1, 1});
  auto std_tensor = torch::tensor(std, options).view({1, 3, 1, 1});

  // Normalize image using vectorized operations
  image_tensor = (image_tensor - mean_tensor) / std_tensor;

  return std::make_tuple(image_tensor, mask_tensor, scale);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
postprocess_predictions(const torch::Tensor &batch_boxes,  // (1,300,4)
                        const torch::Tensor &batch_scores, // (1,300)
                        const torch::Tensor &batch_labels, // (1,300)
                        float scale, float score_threshold = 0.3, float iou_threshold = 0.5) {

  // Batch size must be 1
  assert(batch_boxes.size(0) == 1);
  // Remove batch dimension
  auto boxes = batch_boxes.index({0});   // (300,4)
  auto scores = batch_scores.index({0}); // (300,)
  auto labels = batch_labels.index({0}); // (300,)
  // Apply score threshold
  auto valid_mask = scores > score_threshold;
  auto indices = valid_mask.nonzero().view(-1); // flatten to 1D tensor
  auto valid_boxes = boxes.index_select(0, indices);
  auto valid_scores = scores.index_select(0, indices);
  auto valid_labels = labels.index_select(0, indices);

  // Apply NMS
  auto keep = vision::ops::nms(valid_boxes, valid_scores, iou_threshold);

  valid_boxes = valid_boxes.index_select(0, keep);
  valid_scores = valid_scores.index_select(0, keep);
  valid_labels = valid_labels.index_select(0, keep);

  valid_boxes /= scale;

  return std::make_tuple(valid_boxes, valid_scores, valid_labels);
}

void draw_boxes(cv::Mat &image, const torch::Tensor &boxes, const torch::Tensor &scores, const torch::Tensor &labels) {
  auto boxes_a = boxes.accessor<float, 2>();
  auto scores_a = scores.accessor<float, 1>();
  auto labels_a = labels.accessor<int64_t, 1>();

  const cv::Scalar box_color(0, 0, 255); // Red in BGR
  const int thickness = 2;

  for (int i = 0; i < boxes_a.size(0); ++i) {
    float x1 = boxes_a[i][0];
    float y1 = boxes_a[i][1];
    float x2 = boxes_a[i][2];
    float y2 = boxes_a[i][3];

    // Draw red rectangle
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color, thickness);

    // Format label text: "class 12: 90.5"
    std::ostringstream oss;
    int label_id = static_cast<int>(labels_a[i]);
    std::string class_label =
        (label_id >= 0 && label_id < coco_class_names.size()) ? coco_class_names[label_id] : "unknown";
    oss << class_label << ": " << std::fixed << std::setprecision(1) << 100.f * scores_a[i];
    std::string label_text = oss.str();

    // Dynamic font scale based on image and box size
    float box_height = y2 - y1;
    float image_height = static_cast<float>(image.rows);

    // Compute relative size of box wrt image
    float rel_height = box_height / image_height;

    // Scale font linearly between 0.4 and 1.0
    float font_scale = std::clamp(0.2f + rel_height * 1.0f, 0.2f, 0.8f);
    int baseline = 0;
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    int font_thickness = 1;

    // Get text size for background box
    cv::Size text_size = cv::getTextSize(label_text, font_face, font_scale, font_thickness, &baseline);

    // Background for better visibility (optional)
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x1 + text_size.width, y1 + text_size.height + 5),
                  cv::Scalar(0, 0, 255), cv::FILLED);

    // Draw text in white
    cv::putText(image, label_text, cv::Point(x1, y1 + text_size.height + 3), font_face, font_scale,
                cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
  }
}

class Logger : public nvinfer1::ILogger {
public:
  explicit Logger(Severity severity = Severity::kWARNING) : reportableSeverity(severity) {}

  void log(Severity severity, char const *msg) noexcept override {
    // Only log messages that are at or above 'reportableSeverity'
    if (severity <= reportableSeverity) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }

private:
  Severity reportableSeverity;
};

bool load_tensorrt_plugin(std::string &trt_plugin_path, nvinfer1::ILogger &logger) {
  if (!initLibNvInferPlugins(&logger, "")) {
    std::cerr << "Failed to initialize TensorRT plugins." << std::endl;
    return false;
  }
  std::cout << "Looking up plugin registry..." << std::endl;
  auto *registry = getPluginRegistry();
  if (!registry) {
    std::cerr << "Plugin registry is null!" << std::endl;
    return false;
  }
  // // Load the plugin library
  std::cout << "Loading plugin library..." << std::endl;
  auto libHandle = registry->loadLibrary(trt_plugin_path.c_str());
  return true;
}

nvinfer1::ICudaEngine *load_trt_engine(const std::string &engine_path, nvinfer1::ILogger &logger) {
  std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
  // 1) Read engine file from disk
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "Error opening engine file: " << engine_path << std::endl;
    return nullptr;
  }
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(file_size);
  file.read(engine_data.data(), file_size);
  file.close();

  // There should exist a function to read the engine file
  // std::vector<char> engine_data = nvinfer1::readModelFromFile(engine_path);

  // 2. Create runtime & deserialize engine
  std::cout << "Creating TensorRT runtime and deserializing engine..." << std::endl;

  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
  if (!runtime) {
    std::cerr << "Failed to create InferRuntime." << std::endl;
    return nullptr;
  }
  nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), file_size);
  if (!engine) {
    std::cerr << "Failed to deserialize CUDA engine." << std::endl;
  }
  return engine;
}

/**
 * Utility: Convert nvinfer1::Dims to vector<int64_t>
 */
inline std::vector<int64_t> dimsToVector(const nvinfer1::Dims &d) {
  std::vector<int64_t> shape(d.nbDims);
  for (int i = 0; i < d.nbDims; ++i) {
    shape[i] = d.d[i];
  }
  return shape;
}

// Convert TRT DataType to size in bytes
inline size_t elementSize(nvinfer1::DataType dtype) {
  switch (dtype) {
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kINT64:
    return 8;
  case nvinfer1::DataType::kBOOL:
    return 1;
  default:
    throw std::runtime_error("Unsupported TRT DataType in elementSize()");
  }
}

struct TensorInfo {
  std::string name;
  nvinfer1::TensorIOMode mode; // input or output
  nvinfer1::DataType dtype;
  nvinfer1::Dims dims;
  torch::Tensor cpu_data; // host staging, if needed
};

/**
 * Utility to compute the number of bytes for a TRT tensor
 */
size_t tensorInfoNumBytes(const TensorInfo &info) {
  size_t numel = 1;
  for (int i = 0; i < info.dims.nbDims; ++i) {
    numel *= info.dims.d[i];
  }
  return numel * elementSize(info.dtype);
}

/**
 * Utility to create a CPU torch::Tensor of the correct dtype
 * from a Torch tensor which might be on GPU or in a different dtype.
 */
torch::Tensor convertToTRTDtype(const torch::Tensor &t, nvinfer1::DataType trt_dtype) {
  switch (trt_dtype) {
  case nvinfer1::DataType::kFLOAT:
    return t.to(torch::kFloat32).contiguous().cpu();
  case nvinfer1::DataType::kHALF:
    return t.to(torch::kFloat16).contiguous().cpu();
  case nvinfer1::DataType::kINT32:
    return t.to(torch::kInt32).contiguous().cpu();
  case nvinfer1::DataType::kINT64:
    return t.to(torch::kInt64).contiguous().cpu();
  default:
    throw std::runtime_error("Unsupported or unexpected TRT DataType in convertToTRTDtype()");
  }
}

/**
 * Run inference on the given engine with batch_inputs, img_masks
 * that are shaped e.g. [1,3,H,W] and [1,H,W].
 * The engine is expected to produce (boxes, scores, labels).
 *
 * Assume that batch_inputs and img_mask are on CPU
 *
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
run_trt_inference(nvinfer1::ICudaEngine *engine,
                  const torch::Tensor &batch_inputs, // [1,3,H,W]
                  const torch::Tensor &img_masks,    // [1,H,W]
                  int benchmark_iterations = -1      // -1 means no benchmark
) {
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  if (!context) {
    throw std::runtime_error("Failed to create IExecutionContext");
  }

  int nIO = engine->getNbIOTensors();
  std::cout << "Engine has " << nIO << " I/O tensors." << std::endl;

  // We'll store device pointers + metadata for each I/O tensor
  // i=0 => batch_inputs
  // i=1 => img_masks
  // i=2 => boxes
  // i=3 => scores
  // i=4 => labels
  std::vector<void *> deviceBuffers(nIO, nullptr);
  std::vector<TensorInfo> infos(nIO);

  for (int i = 0; i < nIO; ++i) {
    // double check whether getTensorName, and getTensorMode exists
    const char *tName = engine->getIOTensorName(i);
    infos[i].name = tName;
    infos[i].mode = engine->getTensorIOMode(tName);
    if (i < 2) {
      assert(infos[i].mode == nvinfer1::TensorIOMode::kINPUT);
    } else {
      assert(infos[i].mode == nvinfer1::TensorIOMode::kOUTPUT);
    }
    infos[i].dtype = engine->getTensorDataType(tName);
    infos[i].dims = engine->getTensorShape(tName);

    // For each I/O, allocate GPU memory
    size_t totalBytes = tensorInfoNumBytes(infos[i]);
    cudaMalloc(&deviceBuffers[i], totalBytes);
    // Bind device pointer:
    context->setTensorAddress(tName, deviceBuffers[i]);
  }

  // convert inputs to TensorRT type (e.g. float16)
  // i=0 => batch_inputs
  auto input0_host = convertToTRTDtype(batch_inputs, infos[0].dtype);
  // i=1 => img_masks
  auto input1_host = convertToTRTDtype(img_masks, infos[1].dtype);
  // Copy inputs => device
  {
    size_t in0_bytes = tensorInfoNumBytes(infos[0]);
    cudaMemcpy(deviceBuffers[0], input0_host.data_ptr(), in0_bytes, cudaMemcpyHostToDevice);
    size_t in1_bytes = tensorInfoNumBytes(infos[0]);
    cudaMemcpy(deviceBuffers[1], input1_host.data_ptr(), in1_bytes, cudaMemcpyHostToDevice);
  }

  std::cout << "Running inference..." << std::endl;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Expect batch size 1
  bool ok = context->enqueueV3(stream);
  if (!ok) {
    throw std::runtime_error("TensorRT executeV3 failed!");
  }
  cudaStreamSynchronize(stream);

  if (benchmark_iterations > 0) {
    std::cout << "Benchmarking over " << benchmark_iterations << " iterations..." << std::endl;
    double total_time_ms = 0.0;
    for (int i = 0; i < benchmark_iterations; ++i) {
      auto start_bench = std::chrono::high_resolution_clock::now();
      context->enqueueV3(stream);
      cudaStreamSynchronize(stream);
      auto end_bench = std::chrono::high_resolution_clock::now();
      double bench_duration = std::chrono::duration<double, std::milli>(end_bench - start_bench).count();
      total_time_ms += bench_duration;
    }
    double avg_time_ms = total_time_ms / benchmark_iterations;
    std::cout << "Average inference time: " << std::fixed << std::setprecision(2) << avg_time_ms << "ms" << std::endl;
  }
  cudaStreamDestroy(stream);

  // We'll assume i=2 => boxes, i=3 => scores, i=4 => labels
  auto out_boxes =
      torch::empty(dimsToVector(infos[2].dims), torch::TensorOptions().device(torch::kCPU).dtype(input0_host.dtype()));
  auto out_scores =
      torch::empty(dimsToVector(infos[3].dims), torch::TensorOptions().device(torch::kCPU).dtype(input0_host.dtype()));
  auto out_labels =
      torch::empty(dimsToVector(infos[4].dims), torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));

  auto num_bytes_boxes = tensorInfoNumBytes(infos[2]);
  auto num_bytes_scores = tensorInfoNumBytes(infos[3]);
  auto num_bytes_labels = tensorInfoNumBytes(infos[4]);

  cudaError_t err;
  err = cudaMemcpy(out_boxes.data_ptr(), deviceBuffers[2], num_bytes_boxes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy for boxes failed: " << cudaGetErrorString(err) << std::endl;
  }
  err = cudaMemcpy(out_scores.data_ptr(), deviceBuffers[3], num_bytes_scores, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy for scores failed: " << cudaGetErrorString(err) << std::endl;
  }
  err = cudaMemcpy(out_labels.data_ptr(), deviceBuffers[4], num_bytes_labels, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "cudaMemcpy for labels failed: " << cudaGetErrorString(err) << std::endl;
  }

  auto out_boxes_fp32 = out_boxes.to(torch::kFloat32);
  auto out_scores_fp32 = out_scores.to(torch::kFloat32);

  for (int i = 0; i < nIO; ++i) {
    cudaFree(deviceBuffers[i]);
  }

  return std::make_tuple(out_boxes_fp32, out_scores_fp32, out_labels);
}

// Helper function to check if a string ends with a specific suffix
bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char *argv[]) {
  // Parse command line arguments
  argparse::ArgumentParser program("codetr_inference");

  program.add_argument("--model")
      .help("Path to the TorchScript model file")
      .default_value(std::string("/home/bryan/expr/co-detr/export/codetr_fp16/codetr.ts"));

  program.add_argument("--input").help("Path to the input image").default_value(std::string("assets/demo.jpg"));

  program.add_argument("--output").help("Path to save the output image").default_value(std::string("output.jpg"));

  program.add_argument("--dtype")
      .help("Data type for inference (float16 or float32). The model must be "
            "exported with the same dtype.")
      .default_value(std::string("float16"));

  program.add_argument("--target-height")
      .help("Target height for input image resizing")
      .default_value(768)
      .scan<'i', int>();

  program.add_argument("--target-width")
      .help("Target width for input image resizing")
      .default_value(1152)
      .scan<'i', int>();

  program.add_argument("--score-threshold")
      .help("Confidence threshold for detection filtering")
      .default_value(0.3f)
      .scan<'f', float>();

  program.add_argument("--iou-threshold")
      .help("IoU threshold for non-maximum suppression")
      .default_value(0.5f)
      .scan<'f', float>();

  program.add_argument("--trt-plugin-path")
      .help("Path to libdeformable_attention_plugin.so")
      .default_value(std::string("/home/bryan/src/Co-DETR-TensorRT/codetr/csrc/"
                                 "build/libdeformable_attention_plugin.so"));

  program.add_argument("--benchmark-iterations")
      .help("Number of times to run model inference for benchmarking")
      .default_value(10)
      .scan<'i', int>();

  // Add argument for TensorRT logging verbosity
  program.add_argument("--trt-verbosity")
      .help("TensorRT logging verbosity: 'warning', 'info', or 'verbose'")
      .default_value(nvinfer1::ILogger::Severity::kWARNING)
      .action([](const std::string &value) {
        static const std::unordered_map<std::string, nvinfer1::ILogger::Severity> severity_map = {
            {"warning", nvinfer1::ILogger::Severity::kWARNING},
            {"info", nvinfer1::ILogger::Severity::kINFO},
            {"verbose", nvinfer1::ILogger::Severity::kVERBOSE}};
        auto it = severity_map.find(value);
        if (it == severity_map.end()) {
          throw std::runtime_error("Invalid TensorRT verbosity level: " + value);
        }
        return it->second;
      });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return EXIT_FAILURE;
  }

  // Get arguments
  std::string model_path = program.get<std::string>("--model");
  std::string image_path = program.get<std::string>("--input");
  std::string output_path = program.get<std::string>("--output");
  std::string dtype_str = program.get<std::string>("--dtype");
  std::string trt_plugin_path = program.get<std::string>("--trt-plugin-path");
  int target_height = program.get<int>("--target-height");
  int target_width = program.get<int>("--target-width");
  float score_threshold = program.get<float>("--score-threshold");
  float iou_threshold = program.get<float>("--iou-threshold");
  int benchmark_iterations = program.get<int>("--benchmark-iterations");

  // Parse the argument
  nvinfer1::ILogger::Severity trt_severity = program.get<nvinfer1::ILogger::Severity>("--trt-verbosity");

  // Pass the severity to the Logger
  Logger logger(trt_severity);

  // Validate dtype
  if (dtype_str != "float16" && dtype_str != "float32") {
    std::cerr << "Error: dtype must be either 'float16' or 'float32'" << std::endl;
    return EXIT_FAILURE;
  }
  // Convert dtype string to torch dtype
  torch::Dtype dtype = (dtype_str == "float16") ? torch::kFloat16 : torch::kFloat32;

  if (!load_tensorrt_plugin(trt_plugin_path, logger)) {
    std::cerr << "Failed to load TensorRT plugin." << std::endl;
    return EXIT_FAILURE;
  }

  bool is_tensorrt_engine = false;
  // Check if the model file is TorchScript or TensorRT Engine
  if (ends_with(model_path, ".ts")) {
    std::cout << "Model is likely TorchScript" << std::endl;
  } else if (ends_with(model_path, ".engine")) {
    std::cout << "Model is likely a Serialized TensorRT Engine" << std::endl;
    is_tensorrt_engine = true;
  } else {
    std::cerr << "Error: Unsupported model file extension. Expected '.ts' or "
                 "'.engine'"
              << std::endl;
    return EXIT_FAILURE;
  }

  // Check CUDA availability
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available!" << std::endl;
    return EXIT_FAILURE;
  }

  torch_tensorrt::set_device(0);

  // Load image
  std::cout << "Loading image from: " << image_path << std::endl;
  // the image will be loaded in BGR format, which is OpenCV's default
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  // Preprocess image
  auto [batch_inputs_fp32, img_masks_fp32, scale] = preprocess_image(image, target_height, target_width);
  std::cout << "Preprocessed image shape: " << batch_inputs_fp32.sizes() << std::endl;
  std::cout << "Image mask shape: " << img_masks_fp32.sizes() << std::endl;
  std::cout << "Scale: " << scale << std::endl;
  // Validate preprocess_image output
  if (!batch_inputs_fp32.defined() || !img_masks_fp32.defined()) {
    std::cerr << "Error: preprocess_image returned undefined tensors!" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Preprocessed image tensors defined." << std::endl;
  torch::Tensor batch_inputs = batch_inputs_fp32.to(dtype);
  torch::Tensor img_masks = img_masks_fp32.to(dtype);
  std::cout << "converted types" << std::endl;
  torch::Tensor boxes, scores, labels;
  if (is_tensorrt_engine) {
    nvinfer1::ICudaEngine *engine = load_trt_engine(model_path, logger);
    if (!engine) {
      return EXIT_FAILURE;
    }
    std::tie(boxes, scores, labels) = run_trt_inference(engine, batch_inputs, img_masks, benchmark_iterations);

  } else {
    // Load model
    std::cout << "Loading model from: " << model_path << std::endl;

    // Load the model using torch::jit::load with CUDA support
    torch::jit::script::Module model = torch::jit::load(model_path, torch::kCUDA);
    model.to(torch::kCUDA);
    model.to(dtype);
    model.eval();
    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_inputs.to(torch::kCUDA));
    inputs.push_back(img_masks.to(torch::kCUDA));
    // Co-DETR model. We assume batch size is 1.
    // The model expects two inputs:
    //     batch_inputs (Tensor): has shape (1, 3, H, W) RGB ordered channels
    //     img_masks (Tensor): masks for the input image, has shape (1, H, W).
    // Output tensors: num_boxes is typically 300
    // boxes: has shape (1,num_boxes,4) where 4 is (x1,y1,x2,y2)
    // scores: has shape (1,num_boxes)
    // labels: has shape (1,num_boxes)

    std::cout << "Running inference..." << std::endl;
    // Also counts as warm-up for benchmarking
    auto output = model.forward(inputs).toTuple();

    if (benchmark_iterations > 0) {
      // Benchmark the model
      std::cout << "Benchmarking over " << benchmark_iterations << " iterations..." << std::endl;
      double total_time_ms = 0.0;
      for (int i = 0; i < benchmark_iterations; ++i) {
        auto start_bench = std::chrono::high_resolution_clock::now();
        model.forward(inputs);
        auto end_bench = std::chrono::high_resolution_clock::now();
        double bench_duration = std::chrono::duration<double, std::milli>(end_bench - start_bench).count();
        total_time_ms += bench_duration;
      }

      double avg_time_ms = total_time_ms / benchmark_iterations;
      std::cout << "Average inference time: " << std::fixed << std::setprecision(2) << avg_time_ms << "ms" << std::endl;
    }
    boxes = output->elements()[0].toTensor().to(torch::kFloat32).cpu();
    scores = output->elements()[1].toTensor().to(torch::kFloat32).cpu();
    labels = output->elements()[2].toTensor().to(torch::kInt64).cpu();
  }

  std::cout << "Postprocessing detections..." << std::endl;
  // Postprocess predictions
  auto [final_boxes, final_scores, final_labels] =
      postprocess_predictions(boxes, scores, labels, scale, score_threshold, iou_threshold);

  // Draw boxes on original image
  draw_boxes(image, final_boxes, final_scores, final_labels);

  // Save output image
  cv::imwrite(output_path, image);
  std::cout << "Output saved to: " << output_path << std::endl;
  return EXIT_SUCCESS;
}