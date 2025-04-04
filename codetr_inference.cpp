#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/ops/nms.h>
#include <torch_tensorrt/torch_tensorrt.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <argparse/argparse.hpp>

std::tuple<torch::Tensor, torch::Tensor, float> preprocess_image(const cv::Mat& image, int target_height, int target_width) {
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
    torch::Tensor image_tensor = torch::from_blob(float_image.data, {target_height, target_width, 3}, options);
    image_tensor = image_tensor.permute({2, 0, 1}); // HWC -> CHW
    image_tensor = image_tensor.unsqueeze(0); // Add batch dimension
    
    torch::Tensor mask_tensor = torch::from_blob(mask.data, {1, target_height, target_width}, options);

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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> postprocess_predictions(
    const torch::Tensor& boxes, // (1,300,4)
    const torch::Tensor& scores, // (1,300)
    const torch::Tensor& labels, // (1,300)
    float scale,
    float score_threshold = 0.3,
    float iou_threshold = 0.5) {
    
    // Apply score threshold
    auto valid_mask = scores > score_threshold;
    auto valid_boxes = boxes.index_select(0, valid_mask.nonzero().squeeze());
    auto valid_scores = scores.index_select(0, valid_mask.nonzero().squeeze());
    auto valid_labels = labels.index_select(0, valid_mask.nonzero().squeeze());
    
    // Apply NMS
    auto keep = vision::ops::nms(valid_boxes, valid_scores, iou_threshold);
    
    valid_boxes = valid_boxes.index_select(0, keep);
    valid_scores = valid_scores.index_select(0, keep);
    valid_labels = valid_labels.index_select(0, keep);

    valid_boxes /= scale;

    return std::make_tuple(valid_boxes, valid_scores, valid_labels);
}

void draw_boxes(cv::Mat& image, const torch::Tensor& boxes, const torch::Tensor& scores, const torch::Tensor& labels) {
    auto boxes_a = boxes.accessor<float, 2>();
    auto scores_a = scores.accessor<float, 1>();
    auto labels_a = labels.accessor<int64_t, 1>();
    
    for (int i = 0; i < boxes_a.size(0); ++i) {
        float x1 = boxes_a[i][0];
        float y1 = boxes_a[i][1];
        float x2 = boxes_a[i][2];
        float y2 = boxes_a[i][3];
        
        // Draw rectangle
        cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        
        // Add label and score
        std::string label_text = "Class " + std::to_string(labels_a[i]) + 
                                " (" + std::to_string(scores_a[i]) + ")";
        cv::putText(image, label_text, cv::Point(x1, y1 - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        argparse::ArgumentParser program("codetr_inference");
        
        program.add_argument("--model")
            .help("Path to the TorchScript model file")
            .default_value(std::string("/home/bryan/expr/co-detr/export/codetr_fp16/codetr.ts"));
            
        program.add_argument("--input")
            .help("Path to the input image")
            .default_value(std::string("assets/demo.jpg"));
            
        program.add_argument("--output")
            .help("Path to save the output image")
            .default_value(std::string("output.jpg"));
            
        program.add_argument("--dtype")
            .help("Data type for inference (float16 or float32)")
            .default_value(std::string("float16"));
            
        try {
            program.parse_args(argc, argv);
        } catch (const std::runtime_error& err) {
            std::cerr << err.what() << std::endl;
            std::cerr << program;
            return 1;
        }
        
        // Get arguments
        std::string model_path = program.get<std::string>("--model");
        std::string image_path = program.get<std::string>("--input");
        std::string output_path = program.get<std::string>("--output");
        std::string dtype_str = program.get<std::string>("--dtype");
        
        // Validate dtype
        if (dtype_str != "float16" && dtype_str != "float32") {
            std::cerr << "Error: dtype must be either 'float16' or 'float32'" << std::endl;
            return 1;
        }
        torch_tensorrt::set_device(0);
        
        // Convert dtype string to torch dtype
        torch::Dtype dtype = (dtype_str == "float16") ? torch::kFloat16 : torch::kFloat32;
        
        // Load model
        std::cout << "Loading model from: " << model_path << std::endl;
        
        // Load the model using torch::jit::load with CUDA support
        torch::jit::script::Module model = torch::jit::load(model_path, torch::kCUDA);
        model.to(torch::kCUDA);
        model.to(dtype);
        model.eval();
        
        // Load image
        std::cout << "Loading image from: " << image_path << std::endl;
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            return 1;
        }
        
        // Preprocess image
        int target_height = 768;
        int target_width = 1152;
        auto [input_tensor, mask_tensor, scale] = preprocess_image(image, target_height, target_width);
        input_tensor = input_tensor.to(torch::kCUDA).to(dtype);
        mask_tensor = mask_tensor.to(torch::kCUDA).to(dtype);
        
        // Run inference
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        inputs.push_back(mask_tensor);
        
        auto output = model.forward(inputs).toTuple();
        auto boxes = output->elements()[0].toTensor().cpu();
        auto scores = output->elements()[1].toTensor().cpu();
        auto labels = output->elements()[2].toTensor().cpu();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
        
        // Postprocess predictions
        auto [final_boxes, final_scores, final_labels] = postprocess_predictions(boxes, scores, labels, scale);
        
        // Draw boxes on original image
        draw_boxes(image, final_boxes, final_scores, final_labels);
        
        // Save output image
        cv::imwrite(output_path, image);
        std::cout << "Output saved to: " << output_path << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 