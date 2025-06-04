#include "layer.h"
#include <stdexcept>
#include <string>

__global__ void relu_kernel(float *x, float *y, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    y[tid] = fmax(0.0f, x[tid]);
  }
  //printf("ReLU: y[%d] = %f\n", tid, y[tid]);
};

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

// Compute sum of exp(x_i - max) using reduction
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

__global__ void softmax_kernel(const float *x, float *y, float max_val,
                               float sum, int len) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) {
    y[tid] = exp(x[tid] - max_val) / sum;
  }
}

void apply_activation(float *d, float *y, int len, int act) {
  size_t grid_size = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
  switch (act) {
  case 1:
    //printf("\n------ ReLU ------\n");
    relu_kernel<<<grid_size, BLOCK_SIZE>>>(d, y, len);
    break;
  case 0: {
    //printf("\n------ Softmax ------\n");
    float *temp_max, *temp_sum;
    cudaMalloc(&temp_max, grid_size * sizeof(float));
    cudaMalloc(&temp_sum, grid_size * sizeof(float));

    // find global maximum
    find_max<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d, temp_max,
                                                                    len);

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