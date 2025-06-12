#include "kernels.h"

#define BLOCK_SIZE 256
#define TILE_SIZE 16

__global__ void conv_forward_kernel_batch_dim_1(
    const real_t *input, // Input feature map [C_in, H_in, W_in]
    const real_t
        *weights,       // Weight tensor [C_out, C_in, kernel_size, kernel_size]
    const real_t *bias, // Bias tensor [C_out] (can be nullptr if no bias)
    real_t *output,     // Output feature map [C_out, H_out, W_out]
    size_t C_in,        // Input channels
    size_t C_out,       // Output channels
    size_t H_in, size_t W_in,   // Input height and width
    size_t H_out, size_t W_out, // Output height and width
    size_t kernel_size,         // Kernel size (assuming square kernel)
    size_t padding              // Padding size
) {
  // Thread and block indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Calculate output position for this thread
  int out_x = bx * TILE_SIZE + tx;
  int out_y = by * TILE_SIZE + ty;

  // Early exit if outside output bounds
  if (out_x >= W_out || out_y >= H_out)
    return;

  // Shared memory for input tile
  // Size needs to accommodate kernel overlap: TILE_SIZE + kernel_size - 1
  int SHARED_SIZE = TILE_SIZE + kernel_size - 1;
  extern __shared__ real_t shared_input[];

  // Process each output channel
  for (int c_out = 0; c_out < C_out; c_out++) {
    float result = 0.0f;

    // Process each input channel
    for (int c_in = 0; c_in < C_in; c_in++) {
      // Load input data into shared memory with padding handling
      // Each thread loads multiple elements to fill the shared memory tile
      for (int load_y = ty; load_y < SHARED_SIZE; load_y += blockDim.y) {
        for (int load_x = tx; load_x < SHARED_SIZE; load_x += blockDim.x) {
          // Calculate input coordinates with padding offset
          int in_x = bx * TILE_SIZE + load_x - padding;
          int in_y = by * TILE_SIZE + load_y - padding;

          // Handle boundary conditions (zero-padding)
          if (in_x >= 0 && in_x < W_in && in_y >= 0 && in_y < H_in) {
            int input_idx = c_in * H_in * W_in + in_y * W_in + in_x;
            shared_input[load_y * SHARED_SIZE + load_x] = input[input_idx];
          } else {
            shared_input[load_y * SHARED_SIZE + load_x] = 0.0f; // Zero padding
          }
        }
      }

      // Synchronize to ensure all threads have loaded their data
      __syncthreads();

      // Perform convolution using shared memory
      for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
          // Index into shared memory (accounting for padding offset)
          int shared_x = tx + kx;
          int shared_y = ty + ky;

          // Weight index: [c_out, c_in, ky, kx]
          int weight_idx = c_out * C_in * kernel_size * kernel_size +
                           c_in * kernel_size * kernel_size + ky * kernel_size +
                           kx;

          // Accumulate the convolution result
          result += shared_input[shared_y * SHARED_SIZE + shared_x] *
                    weights[weight_idx];
        }
      }

      // Synchronize before loading next input channel
      __syncthreads();
    }

    // Add bias if provided
    if (bias != nullptr) {
      result += bias[c_out];
    }

    // Store result to output
    int output_idx = c_out * H_out * W_out + out_y * W_out + out_x;
    output[output_idx] = result;
  }
}

void calculate_output_dimensions(size_t H_in, size_t W_in, size_t kernel_size,
                                 size_t padding, size_t &H_out, size_t &W_out) {
  // for stride = 1:
  H_out = H_in + 2 * padding - kernel_size + 1;
  W_out = W_in + 2 * padding - kernel_size + 1;
}

void conv2d_forward(const real_t *input, const real_t *weights,
                    const real_t *bias, real_t *output, size_t batch_size,
                    size_t input_channels, size_t output_channels,
                    size_t kernel_h, size_t kernel_w) {

  size_t padding = 1; // TODO: change it
  size_t H_out, W_out;
  size_t H_in = 28; // TODO: change it
  size_t W_in = 28; // TODO: change it
  calculate_output_dimensions(H_in, W_in, kernel_h, padding, H_out, W_out);

  dim3 blockDim(TILE_SIZE, TILE_SIZE);
  dim3 gridDim((W_out + TILE_SIZE - 1) / TILE_SIZE,
               (H_out + TILE_SIZE - 1) / TILE_SIZE);

  // Calculate shared memory size
  int SHARED_SIZE = TILE_SIZE + kernel_h - 1;
  size_t shared_mem_size = SHARED_SIZE * SHARED_SIZE * sizeof(real_t);

  // Launch kernel
  if (batch_size == 1) {
    conv_forward_kernel_batch_dim_1<<<gridDim, blockDim, shared_mem_size>>>(
        input, weights, bias, output, input_channels, output_channels, H_in,
        W_in, H_out, W_out, kernel_h, padding);
  } else {
    throw std::runtime_error(
        "Batch size > 1 not implemented for conv2d_forward");
  }

  // Check for kernel launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Conv2D kernel launch error: %s\n", cudaGetErrorString(err));
  }
}
