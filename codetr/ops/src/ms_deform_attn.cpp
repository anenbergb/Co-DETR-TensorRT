#include <torch/extension.h>

// CUDA forward declaration
torch::Tensor ms_deform_attn_forward(const torch::Tensor &value,
                                     const torch::Tensor &spatial_shapes,
                                     const torch::Tensor &level_start_index,
                                     const torch::Tensor &sampling_loc,
                                     const torch::Tensor &attn_weight,
                                     const int im2col_step);

// Binding for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "MultiScale Deformable Attention forward (CUDA)");
}
