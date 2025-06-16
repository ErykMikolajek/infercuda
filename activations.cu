/**
 * @file activations.cu
 * @brief CUDA implementation of neural network activation functions
 *
 * This file contains CUDA kernel implementations for various activation
 * functions including ReLU and Softmax, along with supporting reduction
 * operations.
 */

#include "layer.h"
#include <stdexcept>
#include <string>

/**
 * @brief CUDA kernel for ReLU activation function
 *
 * Applies the ReLU activation function: f(x) = max(0, x)
 *
 * @param x Input array
 * @param y Output array
 * @param len Length of the arrays
 */
__global__ void relu_kernel(float *x, float *y, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        y[tid] = fmax(0.0f, x[tid]);
    }
    // printf("ReLU: y[%d] = %f\n", tid, y[tid]);
};

/**
 * @brief CUDA kernel to find maximum value in array
 *
 * Uses parallel reduction to find the maximum value in an array.
 * Each block computes a local maximum, and the results are stored
 * in the max_val array for further processing.
 *
 * @param x Input array
 * @param max_val Array to store block-wise maximum values
 * @param len Length of the input array
 */
__global__ void find_max(const float *x, float *max_val, int len) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < len) ? x[i] : -INFINITY;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        max_val[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief CUDA kernel to compute sum of exponentials
 *
 * Computes the sum of exp(x_i - max_val) using parallel reduction.
 * This is a key step in the softmax computation.
 *
 * @param x Input array
 * @param max_val Maximum value (for numerical stability)
 * @param sum Array to store block-wise sums
 * @param len Length of the input array
 */
__global__ void compute_sum(const float *x, float max_val, float *sum,
                            int len) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < len) ? exp(x[i] - max_val) : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief CUDA kernel for softmax activation function
 *
 * Applies the softmax activation function: f(x_i) = exp(x_i - max) /
 * sum(exp(x_j - max))
 *
 * @param x Input array
 * @param y Output array
 * @param max_val Maximum value (for numerical stability)
 * @param sum Sum of exponentials
 * @param len Length of the arrays
 */
__global__ void softmax_kernel(const float *x, float *y, float max_val,
                               float sum, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        y[tid] = exp(x[tid] - max_val) / sum;
    }
}

/**
 * @brief Host function to apply activation functions
 *
 * This function handles the CUDA kernel launch for various activation
 * functions. Currently supports:
 * - ReLU (act = 1)
 * - Softmax (act = 0)
 *
 * For softmax, it performs a three-step process:
 * 1. Find the maximum value for numerical stability
 * 2. Compute the sum of exponentials
 * 3. Apply the final softmax transformation
 *
 * @param d Input array on device
 * @param y Output array on device
 * @param len Length of the arrays
 * @param act Activation function type (0: Softmax, 1: ReLU)
 * @throws std::runtime_error if activation function type is invalid
 */
void apply_activation(float *d, float *y, int len, int act) {
    size_t grid_size = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    switch (act) {
    case 1:
        // printf("\n------ ReLU ------\n");
        relu_kernel<<<grid_size, BLOCK_SIZE>>>(d, y, len);
        break;
    case 0: {
        // printf("\n------ Softmax ------\n");
        float *temp_max, *temp_sum;
        cudaMalloc(&temp_max, grid_size * sizeof(float));
        cudaMalloc(&temp_sum, grid_size * sizeof(float));

        // find global maximum
        find_max<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d, temp_max, len);

        // If we have multiple blocks, we need a final reduction
        float h_max = -INFINITY;
        if (grid_size > 1) {
            float *h_temp_max = new float[grid_size];
            cudaMemcpy(h_temp_max, temp_max, grid_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < grid_size; i++) {
                h_max = fmax(h_max, h_temp_max[i]);
            }
            delete[] h_temp_max;
        } else {
            cudaMemcpy(&h_max, temp_max, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Compute sum of exp(x_i - max)
        compute_sum<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d, h_max, temp_sum, len);

        float h_sum = 0.0f;
        if (grid_size > 1) {
            float *h_temp_sum = new float[grid_size];
            cudaMemcpy(h_temp_sum, temp_sum, grid_size * sizeof(float),
                       cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < grid_size; i++) {
                h_sum += h_temp_sum[i];
            }

            delete[] h_temp_sum;
        } else {
            cudaMemcpy(&h_sum, temp_sum, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Apply final softmax transformation
        softmax_kernel<<<grid_size, BLOCK_SIZE>>>(d, y, h_max, h_sum, len);

        cudaFree(temp_max);
        cudaFree(temp_sum);
        break;
    }
    default:
        throw std::runtime_error("Invalid activation function");
    }
};