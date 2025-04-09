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
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic_light", "fire_hydrant", "stop_sign",
    "parking_meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports_ball",
    "kite",          "baseball_bat", "baseball_glove",
    "skateboard",    "surfboard",    "tennis_racket",
    "bottle",        "wine_glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot_dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted_plant", "bed",
    "dining_table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell_phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy_bear",
    "hair_drier",    "toothbrush"};

std::tuple<torch::Tensor, torch::Tensor, float>
preprocess_image(const cv::Mat &image, int target_height, int target_width) {
  // Convert BGR to RGB
  cv::Mat rgb_image;
  cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

  // Resize while maintaining aspect ratio
  int orig_height = rgb_image.rows;
  int orig_width = rgb_image.cols;
  float scale = std::min(static_cast<float>(target_width) / orig_width,
                         static_cast<float>(target_height) / orig_height);

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
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor image_tensor = torch::from_blob(
      float_image.data, {target_height, target_width, 3}, options);
  image_tensor = image_tensor.permute({2, 0, 1}); // HWC -> CHW
  image_tensor = image_tensor.unsqueeze(0);       // Add batch dimension

  torch::Tensor mask_tensor =
      torch::from_blob(mask.data, {1, target_height, target_width}, options);

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
                        float scale, float score_threshold = 0.3,
                        float iou_threshold = 0.5) {

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

void draw_boxes(cv::Mat &image, const torch::Tensor &boxes,
                const torch::Tensor &scores, const torch::Tensor &labels) {
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
    cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), box_color,
                  thickness);

    // Format label text: "class 12: 90.5"
    std::ostringstream oss;
    int label_id = static_cast<int>(labels_a[i]);
    std::string class_label =
        (label_id >= 0 && label_id < coco_class_names.size())
            ? coco_class_names[label_id]
            : "unknown";
    oss << class_label << ": " << std::fixed << std::setprecision(1)
        << 100.f * scores_a[i];
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
    cv::Size text_size = cv::getTextSize(label_text, font_face, font_scale,
                                         font_thickness, &baseline);

    // Background for better visibility (optional)
    cv::rectangle(image, cv::Point(x1, y1),
                  cv::Point(x1 + text_size.width, y1 + text_size.height + 5),
                  cv::Scalar(0, 0, 255), cv::FILLED);

    // Draw text in white
    cv::putText(image, label_text, cv::Point(x1, y1 + text_size.height + 3),
                font_face, font_scale, cv::Scalar(255, 255, 255),
                font_thickness, cv::LINE_AA);
  }
}

struct Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cout << "[TRT] " << msg << std::endl;
    }
  }
};

bool load_tensorrt_plugin(std::string &trt_plugin_path) {
  Logger logger;

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

nvinfer1::ICudaEngine *load_trt_engine(const std::string &engine_path,
                                       nvinfer1::ILogger &logger) {
  std::cout << "Loading TensorRT engine from: " << engine_path << std::endl;
  // 1. Read engine file into memory
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file.good()) {
    std::cerr << "Error opening engine file: " << engine_path << std::endl;
    return nullptr;
  }
  const size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> engine_data(file_size);
  file.read(engine_data.data(), file_size);
  file.close();

  // There should exist a function to read the engine file
  // std::vector<char> engine_data = nvinfer1::readModelFromFile(engine_path);

  std::cout << "Creating TensorRT runtime and deserializing engine..."
            << std::endl;
  // 2. Create runtime & deserialize engine
  nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
  if (!runtime) {
    std::cerr << "Failed to create InferRuntime." << std::endl;
    return nullptr;
  }
  nvinfer1::ICudaEngine *engine =
      runtime->deserializeCudaEngine(engine_data.data(), file_size);
  if (!engine) {
    std::cerr << "Failed to deserialize CUDA engine." << std::endl;
    runtime->destroy();
    return nullptr;
  }

  // If you only create one engine, you can destroy runtime here
  runtime->destroy();
  return engine;
}

inline size_t volume(const std::vector<int64_t> &dims) {
  size_t v = 1;
  for (auto d : dims)
    v *= d;
  return v;
}

/**
 * Utility: Convert nvinfer1::Dims to vector<int64_t>
 */
std::vector<int64_t> dimsToVector(const nvinfer1::Dims &d) {
  std::vector<int64_t> shape(d.nbDims);
  for (int i = 0; i < d.nbDims; ++i) {
    shape[i] = d.d[i];
  }
  return shape;
}

/**
 * Returns size in bytes for a given TRT data type:
 * - kFLOAT -> 4
 * - kHALF  -> 2
 * - kINT32 -> 4
 * - etc.
 */
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
  case nvinfer1::DataType::kBOOL:
    return 1;
  case nvinfer1::DataType::kINT64:
    return 8;
  default:
    throw std::runtime_error("Unsupported DataType in elementSize()");
  }
}

/**
 * Utility to create a CPU torch::Tensor of the correct dtype
 * from a Torch tensor which might be on GPU or in a different dtype.
 */
torch::Tensor convertToTRTDtype(const torch::Tensor &t,
                                nvinfer1::DataType trt_dtype) {
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
    throw std::runtime_error(
        "Unsupported or unexpected TRT DataType in convertToTRTDtype()");
  }
}

/**
 * Run inference on the given engine with batch_inputs, img_masks
 * that are shaped e.g. [1,3,H,W] and [1,H,W].
 * The engine is expected to produce (boxes, scores, labels).
 *
 * Steps:
 * 1) Create an ExecutionContext
 * 2) Determine binding indices
 * 3) Convert inputs to correct dtype (float32/float16)
 * 4) Allocate device buffers
 * 5) Copy inputs -> GPU
 * 6) context->executeV2
 * 7) Copy outputs -> host
 * 8) Convert to torch::Tensor
 * 9) Cleanup
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
run_trt_inference(nvinfer1::ICudaEngine *engine,
                  const torch::Tensor &batch_inputs, // shape [1,3,H,W]
                  const torch::Tensor &img_masks)    // shape [1,H,W]
{
  // 1. Create execution context
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  if (!context) {
    throw std::runtime_error("Failed to create ExecutionContext!");
  }

  // Suppose your engine has 5 bindings:
  // 0: input0 -> batch_inputs
  // 1: input1 -> img_masks
  // 2: out_boxes
  // 3: out_scores
  // 4: out_labels
  int inputIndex0 = 0;
  int inputIndex1 = 1;
  int outIndex0 = 2;
  int outIndex1 = 3;
  int outIndex2 = 4;

  // By default Co-DETR export doesn't use dynamic shapes
  // but if you do, you can set the binding dimensions here
  // int H = batch_inputs.size(2);
  // int W = batch_inputs.size(3);
  // context->setBindingDimensions(inputIndex0, nvinfer1::Dims4(1,3,H,W));
  // context->setBindingDimensions(inputIndex1, nvinfer1::Dims3(1,H,W));

  // 2. Inspect engine data types
  nvinfer1::DataType in0_dtype = engine->getBindingDataType(inputIndex0);
  nvinfer1::DataType in1_dtype = engine->getBindingDataType(inputIndex1);
  nvinfer1::DataType out0_dtype = engine->getBindingDataType(outIndex0);
  nvinfer1::DataType out1_dtype = engine->getBindingDataType(outIndex1);
  nvinfer1::DataType out2_dtype = engine->getBindingDataType(outIndex2);

  // 3. Convert input Tensors to match TRT dtype
  auto in0_host = convertToTRTDtype(batch_inputs, in0_dtype);
  auto in1_host = convertToTRTDtype(img_masks, in1_dtype);

  // 4. Figure out binding dims (static or dynamic)
  //    For dynamic shape, do it after setBindingDimensions()
  auto bdim_in0 = context->getBindingDimensions(inputIndex0);
  auto bdim_in1 = context->getBindingDimensions(inputIndex1);
  auto bdim_out0 = context->getBindingDimensions(outIndex0);
  auto bdim_out1 = context->getBindingDimensions(outIndex1);
  auto bdim_out2 = context->getBindingDimensions(outIndex2);

  // Convert to vector<int64_t> for Torch
  auto shape_in0 = dimsToVector(bdim_in0);   // e.g. [1,3,H,W]
  auto shape_in1 = dimsToVector(bdim_in1);   // e.g. [1,H,W]
  auto shape_out0 = dimsToVector(bdim_out0); // e.g. [1,300,4]
  auto shape_out1 = dimsToVector(bdim_out1); // e.g. [1,300]
  auto shape_out2 = dimsToVector(bdim_out2); // e.g. [1,300]

  // 5. Compute the memory sizes (in bytes)
  size_t in0_numel = in0_host.numel();
  size_t in1_numel = in1_host.numel();
  size_t in0_bytes = in0_numel * elementSize(in0_dtype);
  size_t in1_bytes = in1_numel * elementSize(in1_dtype);

  // For outputs, we rely on the shape from binding dims
  auto out0_numel = 1ULL;
  for (auto d : shape_out0)
    out0_numel *= d;
  auto out1_numel = 1ULL;
  for (auto d : shape_out1)
    out1_numel *= d;
  auto out2_numel = 1ULL;
  for (auto d : shape_out2)
    out2_numel *= d;

  size_t out0_bytes = out0_numel * elementSize(out0_dtype);
  size_t out1_bytes = out1_numel * elementSize(out1_dtype);
  size_t out2_bytes = out2_numel * elementSize(out2_dtype);

  // 6. Allocate device buffers
  std::vector<void *> deviceBuffers(engine->getNbBindings(), nullptr);
  cudaMalloc(&deviceBuffers[inputIndex0], in0_bytes);
  cudaMalloc(&deviceBuffers[inputIndex1], in1_bytes);
  cudaMalloc(&deviceBuffers[outIndex0], out0_bytes);
  cudaMalloc(&deviceBuffers[outIndex1], out1_bytes);
  cudaMalloc(&deviceBuffers[outIndex2], out2_bytes);

  // 7. Copy inputs to device
  cudaMemcpy(deviceBuffers[inputIndex0], in0_host.data_ptr(), in0_bytes,
             cudaMemcpyHostToDevice);

  cudaMemcpy(deviceBuffers[inputIndex1], in1_host.data_ptr(), in1_bytes,
             cudaMemcpyHostToDevice);

  // 8. Run inference
  context->executeV2(deviceBuffers.data());

  // 9. Copy outputs back
  std::vector<char> out0_host_vec(out0_bytes);
  std::vector<char> out1_host_vec(out1_bytes);
  std::vector<char> out2_host_vec(out2_bytes);

  cudaMemcpy(out0_host_vec.data(), deviceBuffers[outIndex0], out0_bytes,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out1_host_vec.data(), deviceBuffers[outIndex1], out1_bytes,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(out2_host_vec.data(), deviceBuffers[outIndex2], out2_bytes,
             cudaMemcpyDeviceToHost);

  // 10. Convert them to torch::Tensor
  // For now, assume boxes & scores are float, labels are int
  // but it depends on your engine.
  auto out_boxes = torch::empty(
      shape_out0,
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto out_scores = torch::empty(
      shape_out1,
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));
  auto out_labels = torch::empty(
      shape_out2,
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));

  // Copy memory into those Tensors
  // out_boxes
  std::memcpy(out_boxes.data_ptr(), out0_host_vec.data(), out0_bytes);

  // out_scores
  std::memcpy(out_scores.data_ptr(), out1_host_vec.data(), out1_bytes);

  // out_labels
  if (out2_dtype == nvinfer1::DataType::kINT32) {
    std::cout << "TensorRT label output is int32, converting to int64"
              << std::endl;
    auto *int32_ptr = reinterpret_cast<int32_t *>(out2_host_vec.data());
    auto num_labels = out2_numel;
    auto out_data = out_labels.data_ptr<int64_t>();
    for (size_t i = 0; i < num_labels; i++) {
      out_data[i] = static_cast<int64_t>(int32_ptr[i]);
    }
  } else if (out2_dtype == nvinfer1::DataType::kINT64) {
    // direct memcpy
    std::memcpy(out_labels.data_ptr(), out2_host_vec.data(), out2_bytes);
  }

  // 11. Cleanup device buffers
  for (void *buf : deviceBuffers) {
    cudaFree(buf);
  }
  context->destroy();

  return std::make_tuple(out_boxes, out_scores, out_labels);
}

int main(int argc, char *argv[]) {
  // Parse command line arguments
  argparse::ArgumentParser program("codetr_inference");

  program.add_argument("--model")
      .help("Path to the TorchScript model file")
      .default_value(
          std::string("/home/bryan/expr/co-detr/export/codetr_fp16/codetr.ts"));

  program.add_argument("--input")
      .help("Path to the input image")
      .default_value(std::string("assets/demo.jpg"));

  program.add_argument("--output")
      .help("Path to save the output image")
      .default_value(std::string("output.jpg"));

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
  int bench_iterations = program.get<int>("--benchmark-iterations");

  // Validate dtype
  if (dtype_str != "float16" && dtype_str != "float32") {
    std::cerr << "Error: dtype must be either 'float16' or 'float32'"
              << std::endl;
    return EXIT_FAILURE;
  }

  if (!load_tensorrt_plugin(trt_plugin_path)) {
    std::cerr << "Failed to load TensorRT plugin." << std::endl;
    return EXIT_FAILURE;
  }

  bool is_tensorrt_engine = false;
  // Check if the model file is TorchScript or TensorRT Engine
  if (model_path.ends_with(".ts")) {
    std::cout << "We assume the model is TorchScript because it ends with '.ts'"
              << std::endl;
  } else if (model_path.ends_with(".engine")) {
    is_tensorrt_engine = true;
    std::cout << "We assume the model is a Serialized TensorRT Engine because "
                 "it ends with '.engine'"
              << std::endl;
  }

  torch_tensorrt::set_device(0);

  // Convert dtype string to torch dtype
  torch::Dtype dtype =
      (dtype_str == "float16") ? torch::kFloat16 : torch::kFloat32;

  // Load image
  std::cout << "Loading image from: " << image_path << std::endl;
  // the image will be loaded in BGR format, which is OpenCV's default
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return EXIT_FAILURE;
  }

  // Preprocess image
  auto [batch_inputs, img_masks, scale] =
      preprocess_image(image, target_height, target_width);
  batch_inputs = batch_inputs.to(torch::kCUDA).to(dtype);
  img_masks = img_masks.to(torch::kCUDA).to(dtype);

  if (is_tensorrt_engine) {
    nvinfer1::ICudaEngine *engine = load_trt_engine(engine_path, logger);
    if (!engine) {
      return EXIT_FAILURE;
    }
    auto [boxes, scores, labels] =
        run_trt_inference(engine, batch_inputs, img_masks);

  } else {
    // Load model
    std::cout << "Loading model from: " << model_path << std::endl;

    // Load the model using torch::jit::load with CUDA support
    torch::jit::script::Module model =
        torch::jit::load(model_path, torch::kCUDA);
    model.to(torch::kCUDA);
    model.to(dtype);
    model.eval();
    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_inputs);
    inputs.push_back(img_masks);
    // Co-DETR model. We assume batch size is 1.
    // The model expects two inputs:
    //     batch_inputs (Tensor): has shape (1, 3, H, W) RGB ordered channels
    //     img_masks (Tensor): masks for the input image, has shape (1, H, W).
    // Output tensors: num_boxes is typically 300
    // boxes: has shape (1,num_boxes,4) where 4 is (x1,y1,x2,y2)
    // scores: has shape (1,num_boxes)
    // labels: has shape (1,num_boxes)
    std::cout << "Running inference..." << std::endl;
    auto output = model.forward(inputs).toTuple();
    auto boxes = output->elements()[0].toTensor().to(torch::kFloat32).cpu();
    auto scores = output->elements()[1].toTensor().to(torch::kFloat32).cpu();
    auto labels = output->elements()[2].toTensor().to(torch::kInt64).cpu();
  }

  std::cout << "Postprocessing detections..." << std::endl;
  // Postprocess predictions
  auto [final_boxes, final_scores, final_labels] = postprocess_predictions(
      boxes, scores, labels, scale, score_threshold, iou_threshold);

  // Draw boxes on original image
  draw_boxes(image, final_boxes, final_scores, final_labels);

  // Save output image
  cv::imwrite(output_path, image);
  std::cout << "Output saved to: " << output_path << std::endl;

  // Benchmark the model
  // std::cout << "Benchmarking over " << bench_iterations << " iterations..."
  //           << std::endl;
  // double total_time_ms = 0.0;
  // for (int i = 0; i < bench_iterations; ++i) {
  //   auto start_bench = std::chrono::high_resolution_clock::now();
  //   model.forward(inputs);
  //   auto end_bench = std::chrono::high_resolution_clock::now();
  //   double bench_duration =
  //       std::chrono::duration<double, std::milli>(end_bench - start_bench)
  //           .count();
  //   total_time_ms += bench_duration;
  // }

  // double avg_time_ms = total_time_ms / bench_iterations;
  // std::cout << "Average inference time: " << std::fixed <<
  // std::setprecision(2)
  //           << avg_time_ms << "ms" << std::endl;

  return EXIT_SUCCESS;
}