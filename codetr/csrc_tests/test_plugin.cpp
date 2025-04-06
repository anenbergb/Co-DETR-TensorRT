#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <iostream>

int main() {
    std::cout << "Initializing TensorRT plugins..." << std::endl;
    class Logger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING)
                std::cout << "[TRT] " << msg << std::endl;
        }
    } logger;
    
    if (!initLibNvInferPlugins(&logger, "")) {
        std::cerr << "Failed to initialize TensorRT plugins." << std::endl;
        return 1;
    }

    std::cout << "Looking up plugin registry..." << std::endl;
    auto* registry = getPluginRegistry();
    if (!registry) {
        std::cerr << "Plugin registry is null!" << std::endl;
        return 1;
    }

    // Load the plugin library
    std::cout << "Loading plugin library..." << std::endl;
    std::string pluginPath = "/home/bryan/src/Co-DETR-TensorRT/codetr/csrc/build/libdeformable_attention_plugin.so";
    // ../../csrc/build/libdeformable_attention_plugin.so";
    auto libHandle = registry->loadLibrary(pluginPath.c_str());
    if (libHandle == nullptr) {
        std::cerr << "Failed to load plugin library: " << pluginPath << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Successfully loaded plugin library" << std::endl;

    int numCreators = 0;
    auto* base = registry->getAllCreators(&numCreators);
    // auto* base = registry->getPluginCreatorList(&numCreators);
    std::cout << "Found " << numCreators << " registered plugins:" << std::endl;

    
    for (int i = 0; i < numCreators; i++) {
        try {
            auto* creator = dynamic_cast<nvinfer1::IPluginCreator*>(base[i]);
            std::cout << "  " << creator->getPluginName() 
                    << " (v" << creator->getPluginVersion() 
                    << ", ns:" << creator->getPluginNamespace() << ")" << std::endl;
        } catch (const std::bad_cast& e) {
            std::cerr << "Failed to cast plugin creator: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }


    std::cout << "Checking for DeformableAttentionPlugin..." << std::endl;
    const char* pluginName = "DeformableAttentionPlugin";
    const char* pluginVersion = "1";

    // auto* creator = registry->getPluginCreator(pluginName, pluginVersion, "");
    auto* creator = registry->getCreator(pluginName, pluginVersion, "");
    if (creator == nullptr) {
        std::cerr << "Plugin creator for " << pluginName << " not found!" << std::endl;
        return 1;
    }

    std::cout << "Successfully found plugin: " << pluginName << std::endl;
    return 0;
}
