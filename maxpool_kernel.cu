#include "kernels.h"

void __global__ maxpool2d_forward_kernel_batch_dim_1(
    const real_t *input, real_t *output, size_t input_dim, size_t output_dim,
    size_t kernel_h, size_t kernel_w, size_t h_in, size_t w_in, size_t h_out,
    size_t w_out) {

    size_t stride = 1;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < w_out && out_y < h_out) {
        for (int c = 0; c < output_dim; c++) {
            real_t max_val = -FLT_MAX;

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
        }
    }
}

void calculate_output_dimensions(size_t h_in, size_t w_in, size_t kernel_size,
                                 size_t &h_out, size_t &w_out) {
    h_out = h_in / kernel_size;
    w_out = w_in / kernel_size;
}

void maxpool2d_forward(const real_t *input, real_t *output, size_t batch_size,
                       size_t input_dim, size_t output_dim, size_t kernel_h,
                       size_t kernel_w, size_t h_in, size_t w_in) {

    size_t h_out, w_out;
    calculate_output_dimensions(h_in, w_in, kernel_h, h_out, w_out);

    dim3 grid_size((w_out + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (h_out + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (batch_size == 1) {
        maxpool2d_forward_kernel_batch_dim_1<<<grid_size, BLOCK_SIZE>>>(
            input, output, input_dim, output_dim, kernel_h, kernel_w, h_in,
            w_in, h_out, w_out);
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