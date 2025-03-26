#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace codetr_cpp {

// Defines the operators
TORCH_LIBRARY(codetr_cpp, m) {
  m.def("ms_deform_attn_forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor");
}

} // namespace codetr_cpp