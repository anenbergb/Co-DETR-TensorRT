#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace codetr_cpp {

// CUDA forward declaration
torch::Tensor ms_deform_attn_forward(const torch::Tensor &value,
                                     const torch::Tensor &spatial_shapes,
                                     const torch::Tensor &level_start_index,
                                     const torch::Tensor &sampling_loc,
                                     const torch::Tensor &attn_weight,
                                     const int im2col_step);

// // CUDA backward declaration
// void ms_deform_attn_backward(const torch::Tensor &value,
//                              const torch::Tensor &spatial_shapes,
//                              const torch::Tensor &level_start_index,
//                              const torch::Tensor &sampling_loc,
//                              const torch::Tensor &attn_weight,
//                              const torch::Tensor &grad_output,
//                              torch::Tensor &grad_value,
//                              torch::Tensor &grad_sampling_loc,
//                              torch::Tensor &grad_attn_weight,
//                              const int im2col_step);


// Defines the operators
TORCH_LIBRARY(codetr_cpp, m) {
  m.def("ms_deform_attn_forward(Tensor value, Tensor spatial_shapes, Tensor level_start_index, Tensor sampling_loc, Tensor attn_weight, int im2col_step) -> Tensor");
}

// Registers CUDA implementation for ms_deform_attn_forward
TORCH_LIBRARY_IMPL(codetr_cpp, CUDA, m) {
  m.impl("ms_deform_attn_forward", &ms_deform_attn_forward);
}

} // namespace codetr_cpp