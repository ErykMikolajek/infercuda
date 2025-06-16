/**
 * @file fc_kernel.cu
 * @brief CUDA implementation of fully connected layer forward pass
 *
 * This file contains the CUDA kernel implementations for the forward pass
 * of fully connected layers, supporting both single and batch processing.
 */

#include "kernels.h"
#include <stdexcept>
#include <string>

#define BLOCK_SIZE 256

/**
 * @brief CUDA kernel for fully connected layer forward pass with batch
 * processing
 * @param input Input data array
 * @param weights Weight matrix
 * @param bias Bias vector
 * @param output Output data array
 * @param batch_size Number of samples in the batch
 * @param input_dim Dimension of input data
 * @param output_dim Dimension of output data
 */
__global__ void fc_forward_kernel_batch_dim_n(const real_t *input,
                                              const real_t *weights,
                                              const real_t *bias,
                                              real_t *output, int batch_size,
                                              int input_dim, int output_dim) {}

/**
 * @brief CUDA kernel for fully connected layer forward pass with single sample
 * @param input Input data array
 * @param weights Weight matrix
 * @param bias Bias vector
 * @param output Output data array
 * @param input_dim Dimension of input data
 * @param output_dim Dimension of output data
 */
__global__ void fc_forward_kernel_batch_dim_1(const real_t *input,
                                              const real_t *weights,
                                              const real_t *bias,
                                              real_t *output, size_t input_dim,
                                              size_t output_dim) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_dim)
        return;

    real_t acc = 0.0f;
    int weight_offset = out_idx * input_dim;
    for (int i = 0; i < input_dim; ++i) {
        acc += input[i] * weights[weight_offset + i];
    }

    output[out_idx] = acc + bias[out_idx];
    // printf("Forward Output[%d] = %f\n", out_idx, output[out_idx]);
}

/**
 * @brief Host function to launch fully connected layer forward pass
 *
 * This function handles the CUDA kernel launch for the fully connected layer
 * forward pass, supporting both single sample and batch processing modes.
 *
 * @param input Input data array
 * @param weights Weight matrix
 * @param bias Bias vector
 * @param output Output data array
 * @param batch_size Number of samples in the batch
 * @param input_dim Dimension of input data
 * @param output_dim Dimension of output data
 * @throws std::runtime_error if any device pointer is null or kernel launch
 * fails
 */
void fc_forward(const real_t *input, const real_t *weights, const real_t *bias,
                real_t *output, size_t batch_size, size_t input_dim,
                size_t output_dim) {

    if (input == nullptr || weights == nullptr || bias == nullptr ||
        output == nullptr) {
        throw std::runtime_error("One or more device pointers are null");
    }

    printf("\n------ Fc Forward ------\n");

    size_t grid_size = (output_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (batch_size == 1) {
        fc_forward_kernel_batch_dim_1<<<grid_size, BLOCK_SIZE>>>(
            input, weights, bias, output, input_dim, output_dim);
    } else {
        fc_forward_kernel_batch_dim_n<<<grid_size, BLOCK_SIZE>>>(
            input, weights, bias, output, batch_size, input_dim, output_dim);
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to launch fc_kernel: " +
                                 std::string(cudaGetErrorString(err)));
    }
}