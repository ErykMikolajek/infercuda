/**
 * @file maxpool_kernel.cu
 * @brief CUDA implementation of 2D max pooling layer forward pass
 *
 * This file contains the CUDA kernel implementation for the forward pass
 * of 2D max pooling layers, which reduce spatial dimensions by taking
 * the maximum value in each pooling window.
 */

#include "kernels.h"

/**
 * @brief CUDA kernel for 2D max pooling forward pass with single sample
 *
 * This kernel implements the forward pass of a 2D max pooling layer.
 * For each output position, it computes the maximum value within a
 * kernel-sized window in the input feature map.
 *
 * @param input Input feature map
 * @param output Output feature map
 * @param input_dim Number of input channels
 * @param output_dim Number of output channels
 * @param kernel_h Height of the pooling kernel
 * @param kernel_w Width of the pooling kernel
 * @param h_in Input height
 * @param w_in Input width
 * @param h_out Output height
 * @param w_out Output width
 */
void __global__ maxpool2d_forward_kernel_batch_dim_1(
    const real_t *input, real_t *output, size_t input_dim, size_t output_dim,
    size_t kernel_h, size_t kernel_w, size_t h_in, size_t w_in, size_t h_out,
    size_t w_out) {

    size_t stride = 1;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < w_out && out_y < h_out) {
        // printf("MaxPool2D Forward: Processing output pixel (%d, %d)\n",
        // out_y, out_x);
        for (int c = 0; c < output_dim; c++) {
            // printf("MaxPool2D Forward: Processing channel %d\n", c);
            real_t max_val = -REAL_MAX;

            for (int py = 0; py < kernel_h; py++) {
                for (int px = 0; px < kernel_w; px++) {
                    int in_x = out_x * stride + px;
                    int in_y = out_y * stride + py;
                    real_t val =
                        input[in_y * w_in * input_dim + in_x * input_dim + c];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            output[out_y * w_out * output_dim + out_x * output_dim + c] =
                max_val;
            // printf("MaxPool2D Forward: out[%d, %d, %d] = %f\n", out_y, out_x,
            // c, 	max_val);
        }
    }
}

/**
 * @brief Calculate output dimensions for 2D max pooling
 *
 * Computes the output dimensions of a 2D max pooling operation given
 * input dimensions and kernel size.
 *
 * @param h_in Input height
 * @param w_in Input width
 * @param kernel_size Size of the pooling kernel
 * @param h_out [out] Output height
 * @param w_out [out] Output width
 */
void calculate_output_dimensions(size_t h_in, size_t w_in, size_t kernel_size,
                                 size_t &h_out, size_t &w_out) {
    h_out = h_in / kernel_size;
    w_out = w_in / kernel_size;
}

/**
 * @brief Host function to launch 2D max pooling forward pass
 *
 * This function handles the CUDA kernel launch for the 2D max pooling layer
 * forward pass. Currently only supports single sample processing.
 *
 * @param input Input feature map
 * @param output Output feature map
 * @param batch_size Number of samples in the batch (must be 1)
 * @param input_channels Number of input channels
 * @param output_channels Number of output channels
 * @param kernel_h Height of the pooling kernel
 * @param kernel_w Width of the pooling kernel
 * @param h_in Input height
 * @param w_in Input width
 * @throws std::runtime_error if batch_size > 1 or kernel launch fails
 */
void maxpool2d_forward(const real_t *input, real_t *output, size_t batch_size,
                       size_t input_channels, size_t output_channels,
                       size_t kernel_h, size_t kernel_w, size_t h_in,
                       size_t w_in) {

    size_t h_out, w_out;
    calculate_output_dimensions(h_in, w_in, kernel_h, h_out, w_out);

    dim3 grid_size((w_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (h_out + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\n------ MaxPool Forward ------\n");

    if (batch_size == 1) {
        maxpool2d_forward_kernel_batch_dim_1<<<grid_size, BLOCK_SIZE>>>(
            input, output, input_channels, output_channels, kernel_h, kernel_w,
            h_in, w_in, h_out, w_out);
    } else {
        throw std::runtime_error(
            "Batch size > 1 not implemented for maxpool2d_forward");
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA MaxPool2D kernel launch error: %s\n",
               cudaGetErrorString(err));
    }
}