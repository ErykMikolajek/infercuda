#include "kernels.h"
#include <stdexcept>
#include <string>

//__global__ void fc_forward_kernel(const real_t* input, const real_t* weights,
//    const real_t* bias, real_t* output,
//    int batch_size, int input_dim, int output_dim) {
//
//    int tid = blockIdx.x * blockDim.x + threadIdx.x;
//    int total_outputs = batch_size * output_dim;
//
//    if (tid < total_outputs) {
//        int batch_idx = tid / output_dim;
//        int out_idx = tid % output_dim;
//
//        real_t sum = bias[out_idx];
//        for (int i = 0; i < input_dim; i++) {
//            sum += input[batch_idx * input_dim + i] *
//                weights[out_idx * input_dim + i];
//        }
//        output[tid] = sum;
//    }
//}

__global__ void fc_forward_kernel_batch1(const real_t *input, 
                                         const real_t *weights,
                                         const real_t *bias,
                                         real_t* output,
                                         int input_dim, int output_dim) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_dim) return;

    real_t acc = 0.0f;
    int weight_offset = out_idx * input_dim;
    for (int i = 0; i < input_dim; ++i) {
        acc += input[i] * weights[weight_offset + i];
    }

    output[out_idx] = acc + bias[out_idx];
}

void fc_forward(const real_t *input, const real_t *weights,
                const real_t *bias, real_t *output,
                size_t batch_size, size_t input_dim, size_t output_dim) {

    if (input == nullptr || weights == nullptr || bias == nullptr ||
        output == nullptr) {
        throw std::runtime_error("One or more device pointers are null");
    }
    
    int grid_size = (output_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fc_forward_kernel_batch1 << <grid_size, BLOCK_SIZE>> >(
        input, weights, bias, output, input_dim, output_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to launch fc_kernel: " +
            std::string(cudaGetErrorString(err)));
    }
}