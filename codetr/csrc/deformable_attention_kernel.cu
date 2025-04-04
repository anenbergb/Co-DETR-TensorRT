#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/TypeCast.h>

extern at::Tensor ms_deform_attn_forward(
    const at::Tensor& value,
    const at::Tensor& spatial_shapes,
    const at::Tensor& level_start_index,
    const at::Tensor& sampling_loc,
    const at::Tensor& attn_weight,
    const int im2col_step);

template <typename scalar_t>
void multi_scale_deformable_attention_cuda_forward(
    const scalar_t* value_ptr,
    const int64_t* spatial_shapes_ptr,
    const int64_t* level_start_index_ptr,
    const scalar_t* sampling_loc_ptr,
    const scalar_t* attn_weight_ptr,
    scalar_t* output_ptr,
    int bs,
    int num_keys,
    int num_queries,
    int num_heads,
    int dim_per_head,
    int num_levels,
    int num_points,
    int im2col_step,
    cudaStream_t stream)
{
    // This should resolve float -> at::kFloat, and __half -> at::kHalf
    at::ScalarType scalar_type = c10::CppTypeToScalarType<scalar_t>::value;
    auto options = at::TensorOptions().dtype(scalar_type).device(at::kCUDA);
    auto options_long = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);

    // We're telling PyTorch to use externally-managed memory provied by TensorRT 
    at::Tensor value = at::from_blob(
        const_cast<scalar_t*>(value_ptr),
        {bs, num_keys, num_heads, dim_per_head},
        options);

    at::Tensor spatial_shapes = at::from_blob(
        const_cast<int64_t*>(spatial_shapes_ptr),
        {num_levels, 2},
        options_long);

    at::Tensor level_start_index = at::from_blob(
        const_cast<int64_t*>(level_start_index_ptr),
        {num_levels},
        options_long);

    at::Tensor sampling_loc = at::from_blob(
        const_cast<scalar_t*>(sampling_loc_ptr),
        {bs, num_queries, num_heads, num_levels, num_points, 2},
        options);

    at::Tensor attn_weight = at::from_blob(
        const_cast<scalar_t*>(attn_weight_ptr),
        {bs, num_queries, num_heads, num_levels, num_points},
        options);

    at::Tensor output = ms_deform_attn_forward(
        value, spatial_shapes, level_start_index,
        sampling_loc, attn_weight, im2col_step);

    // This memcpy from output to output_ptr can be removed if you pre-allocate the output using from_blob and pass it into ms_deform_attn_forward.
    std::memcpy(output_ptr, output.data_ptr<scalar_t>(), output.numel() * sizeof(scalar_t));
}
