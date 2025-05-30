#include "kernels.h"

__global__ void fc_forward_kernel(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    int batch_size, int input_dim, int output_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * output_dim;

    if (tid < total_outputs) {
        int batch_idx = tid / output_dim;
        int out_idx = tid % output_dim;

        real_t sum = bias[out_idx];
        for (int i = 0; i < input_dim; i++) {
            sum += input[batch_idx * input_dim + i] *
                weights[out_idx * input_dim + i];
        }
        output[tid] = sum;
    }
}

void fc_forward(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    size_t batch_size, size_t input_dim, size_t output_dim) {
    int total_outputs = batch_size * output_dim;
    int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;

    fc_forward_kernel << <grid_size, block_size, 0, stream >> > (
        input, weights, bias, output, batch_size, input_dim, output_dim);
}