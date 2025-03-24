#include <torch/extension.h>
#include <vector>

// CUDA forward declaration
torch::Tensor ms_deformable_attn_forward(
    torch::Tensor query, torch::Tensor value, 
    torch::Tensor spatial_shapes, torch::Tensor level_start_index, 
    torch::Tensor sampling_locations, torch::Tensor attention_weights);

// Binding for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ms_deformable_attn_forward", &ms_deformable_attn_forward, "MultiScale Deformable Attention forward (CUDA)");
}
