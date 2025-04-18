#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  // Check if the plugin path is provided as a command-line argument
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <path_to_libdeformable_attention_plugin.so>" << std::endl;
    return EXIT_FAILURE;
  }

  // Get the plugin path from the command-line argument
  std::string pluginPath = argv[1];

  std::cout << "Initializing TensorRT plugins..." << std::endl;
  class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
      if (severity <= Severity::kWARNING)
        std::cout << "[TRT] " << msg << std::endl;
    }
  } logger;

  if (!initLibNvInferPlugins(&logger, "")) {
    std::cerr << "Failed to initialize TensorRT plugins." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Looking up plugin registry..." << std::endl;
  auto *registry = getPluginRegistry();
  if (!registry) {
    std::cerr << "Plugin registry is null!" << std::endl;
    return EXIT_FAILURE;
  }

  // Load the plugin library
  std::cout << "Loading plugin library from: " << pluginPath << std::endl;
  auto libHandle = registry->loadLibrary(pluginPath.c_str());

  int numCreators = 0;
  auto *base = registry->getAllCreators(&numCreators);
  std::cout << "Found " << numCreators << " registered plugins:" << std::endl;

  std::cout << "Checking for DeformableAttentionPlugin..." << std::endl;
  const char *pluginName = "DeformableAttentionPlugin";
  const char *pluginVersion = "1";

  auto *creator = static_cast<nvinfer1::IPluginCreatorV3One *>(registry->getCreator(pluginName, pluginVersion, ""));
  if (creator == nullptr) {
    std::cerr << "Plugin creator for " << pluginName << " not found!" << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Successfully found plugin: " << pluginName << std::endl;
  return EXIT_SUCCESS;
}
