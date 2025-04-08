#include <ATen/ATen.h>
#include <torch/library.h>

namespace codetr {

// Forward declaration for torch extension
at::Tensor ms_deform_attn_forward(const at::Tensor &value,
                                  const at::Tensor &spatial_shapes,
                                  const at::Tensor &level_start_index,
                                  const at::Tensor &sampling_loc,
                                  const at::Tensor &attn_weight,
                                  const int64_t im2col_step);

void ms_deform_attn_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight, const at::Tensor &grad_output,
    at::Tensor &grad_value, at::Tensor &grad_sampling_loc,
    at::Tensor &grad_attn_weight, const int64_t im2col_step);

TORCH_LIBRARY(codetr, m) {
  m.def("multi_scale_deformable_attention(Tensor value, Tensor spatial_shapes, "
        "Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, "
        "int im2col_step) -> Tensor");
  m.def(
      "multi_scale_deformable_attention_backward(Tensor value, Tensor "
      "spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor "
      "attn_weight, Tensor grad_output, Tensor(a!) grad_value, Tensor(b!) "
      "grad_sampling_loc, Tensor(c!) grad_attn_weight, int im2col_step) -> ()");
}

// Registers CUDA implementation for ms_deform_attn_forward
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func
TORCH_LIBRARY_IMPL(codetr, CUDA, m) {
  m.impl("multi_scale_deformable_attention", &ms_deform_attn_forward);
  m.impl("multi_scale_deformable_attention_backward", &ms_deform_attn_backward);
}

} // namespace codetr