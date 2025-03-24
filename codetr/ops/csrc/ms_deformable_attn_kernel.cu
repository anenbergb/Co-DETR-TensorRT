#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Here's a detailed explanation of how the kernel works:
 *
 * 1. Kernel Launch: The kernel is launched with a grid of threads. The number of threads per block is defined by THREADS_PER_BLOCK, and the number of blocks is calculated based on the total number of combinations of batch index, head index, and query index (num_kernels).
 * 2. Thread Index Calculation: Each thread calculates its global index (idx) using blockIdx.x, blockDim.x, and threadIdx.x. This index uniquely identifies the combination of batch index, head index, and query index that the thread will process.
 * 3. Bounds Check: The thread checks if its index is within the valid range (idx < batch_size * num_heads * num_queries). If not, the thread returns early.
 * 4. Index Decomposition: The thread decomposes its global index (idx) into batch index (b), head index (h), and query index (q).
 * 5. Result Initialization: The thread initializes a variable (result) to accumulate the computed value.
 * 6. Feature Level Loop: The thread loops over each feature level (num_levels).
 * 7. Sampling Point Loop: Within each feature level, the thread loops over each sampling point (num_points).
 * 8. Sampling Location and Attention Weight: The thread retrieves the sampling location (x, y) and attention weight (weight) for the current sampling point.
 * 9. Bilinear Interpolation: The thread performs bilinear interpolation using the sampling location and the value tensor. It calculates the low and high coordinates (x_low, y_low, x_high, y_high) and the interpolation weights (lx, ly, hx, hy). It then computes the interpolated value (val) and accumulates it into result.
 * 10 Store Result: After processing all feature levels and sampling points, the thread stores the accumulated result in the output tensor.
 * The kernel is launched once, and each thread in the grid handles a different combination of batch index, head index, and query index. The number of threads launched is equal to batch_size * num_heads * num_queries.
 * 
 */
// Define the number of threads per block for the CUDA kernel
#define THREADS_PER_BLOCK 256

/**
 * @brief CUDA kernel for multi-scale deformable attention.
 *
 * @param query Input query tensor of shape (batch_size, num_heads, num_queries, embed_dim).
 * @param value Input value tensor of shape (batch_size, num_levels, height, width, embed_dim).
 * @param spatial_shapes Tensor containing spatial shapes of each level, of shape (num_levels, 2).
 * @param level_start_index Tensor containing start index of each level, of shape (num_levels).
 * @param sampling_locations Tensor containing sampling locations, of shape (batch_size, num_heads, num_queries, num_levels, num_points, 2).
 * @param attention_weights Tensor containing attention weights, of shape (batch_size, num_heads, num_queries, num_levels, num_points).
 * @param output Output tensor of shape (batch_size, num_heads, num_queries, embed_dim).
 * @param batch_size Number of batches.
 * @param num_heads Number of attention heads.
 * @param num_levels Number of feature levels.
 * @param num_queries Number of queries.
 * @param num_points Number of sampling points.
 * @param embed_dim Embedding dimension.
 */
__global__ void ms_deformable_attn_kernel(
    const float* query, const float* value,
    const int* spatial_shapes, const int* level_start_index,
    const float* sampling_locations, const float* attention_weights,
    float* output, int batch_size, int num_heads, 
    int num_levels, int num_queries, int num_points, int embed_dim) {

    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // If the index is out of bounds, return
    if (idx >= batch_size * num_heads * num_queries) return;

    // Calculate the batch index
    int b = idx / (num_heads * num_queries);
    // Calculate the head index
    int h = (idx / num_queries) % num_heads;
    // Calculate the query index
    int q = idx % num_queries;

    // Initialize the result to 0
    float result = 0.0f;
    
    // Loop over each feature level
    for (int l = 0; l < num_levels; ++l) {
        // Get the start index for the current level
        int start_index = level_start_index[l];
        // Get the spatial height and width for the current level
        int h_spatial = spatial_shapes[l * 2];
        int w_spatial = spatial_shapes[l * 2 + 1];

        // Loop over each sampling point
        for (int p = 0; p < num_points; ++p) {
            // Calculate the index for the sampling location and attention weight
            int loc_index = (((b * num_heads + h) * num_queries + q) * num_levels + l) * num_points + p;
            int attn_index = loc_index;
            
            // Get the sampling location and attention weight
            float x = sampling_locations[2 * loc_index];
            float y = sampling_locations[2 * loc_index + 1];
            float weight = attention_weights[attn_index];

            // Calculate the low and high coordinates for bilinear interpolation
            int x_low = static_cast<int>(floor(x));
            int y_low = static_cast<int>(floor(y));
            int x_high = min(x_low + 1, w_spatial - 1);
            int y_high = min(y_low + 1, h_spatial - 1);

            // Calculate the interpolation weights
            float lx = x - x_low, ly = y - y_low;
            float hx = 1 - lx, hy = 1 - ly;

            // Calculate the index for the value tensor
            int v_idx = (((b * num_levels + l) * h_spatial + y_low) * w_spatial + x_low) * embed_dim + h;
            // Perform bilinear interpolation and accumulate the result
            float val = value[v_idx] * hx * hy + 
                        value[v_idx + 1] * lx * hy +
                        value[v_idx + w_spatial] * hx * ly +
                        value[v_idx + w_spatial + 1] * lx * ly;

            result += weight * val;
        }
    }

    // Store the result in the output tensor
    output[idx] = result;
}

/**
 * @brief Forward function for multi-scale deformable attention.
 *
 * @param query Input query tensor of shape (batch_size, num_heads, num_queries, embed_dim).
 * @param value Input value tensor of shape (batch_size, num_levels, height, width, embed_dim).
 * @param spatial_shapes Tensor containing spatial shapes of each level, of shape (num_levels, 2).
 * @param level_start_index Tensor containing start index of each level, of shape (num_levels).
 * @param sampling_locations Tensor containing sampling locations, of shape (batch_size, num_heads, num_queries, num_levels, num_points, 2).
 * @param attention_weights Tensor containing attention weights, of shape (batch_size, num_heads, num_queries, num_levels, num_points).
 * @return Output tensor of shape (batch_size, num_heads, num_queries, embed_dim).
 */
torch::Tensor ms_deformable_attn_forward(
    torch::Tensor query, torch::Tensor value, 
    torch::Tensor spatial_shapes, torch::Tensor level_start_index, 
    torch::Tensor sampling_locations, torch::Tensor attention_weights) {

    // Get the dimensions of the input tensors
    const int batch_size = query.size(0);
    const int num_heads = query.size(1);
    const int num_queries = query.size(2);
    const int embed_dim = query.size(3);
    const int num_levels = spatial_shapes.size(0);
    const int num_points = sampling_locations.size(3);

    // Create an output tensor filled with zeros
    auto output = torch::zeros({batch_size, num_heads, num_queries, embed_dim}, query.options());

    // Calculate the number of kernels and blocks for the CUDA kernel launch
    const int num_kernels = batch_size * num_heads * num_queries;
    const int num_blocks = (num_kernels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the CUDA kernel
    ms_deformable_attn_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        query.data_ptr<float>(), value.data_ptr<float>(),
        spatial_shapes.data_ptr<int>(), level_start_index.data_ptr<int>(),
        sampling_locations.data_ptr<float>(), attention_weights.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, num_heads, 
        num_levels, num_queries, num_points, embed_dim);
    
    // Return the output tensor
    return output;
}
