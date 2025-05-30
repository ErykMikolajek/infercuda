#include "kernels.h"

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

__global__ void fc_forward_kernel_batch1(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    int input_dim, int output_dim) {

    extern __shared__ real_t shared_input[];

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < input_dim)
        shared_input[threadIdx.x] = input[threadIdx.x];
    __syncthreads();

    if (out_idx >= output_dim) return;

    real_t acc = 0.0f;
    int weight_offset = out_idx * input_dim;
    for (int i = 0; i < input_dim; ++i) {
        acc += shared_input[i] * weights[weight_offset + i];
    }

    output[out_idx] = acc + bias[out_idx];
}

void fc_forward(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    size_t batch_size, size_t input_dim, size_t output_dim) {

    dim3 grid_size = (output_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = input_dim * sizeof(float);
    fc_forward_kernel_batch1 << <grid_size, BLOCK_SIZE, shared_mem_size >> > (input, weights, bias, output, input_dim, output_dim);

}