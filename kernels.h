/**
 * @file kernels.h
 * @brief CUDA kernel declarations for neural network operations
 *
 * This file contains declarations for CUDA kernels that implement
 * various neural network operations like fully connected layers,
 * convolutional layers, pooling, and activation functions.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "common.h"

/**
 * @brief Forward pass for fully connected layer
 * @param input Input data array
 * @param weights Weight matrix
 * @param bias Bias vector
 * @param output Output data array
 * @param batch_size Number of samples in the batch
 * @param input_dim Dimension of input data
 * @param output_dim Dimension of output data
 */
void fc_forward(const real_t *input, const real_t *weights, const real_t *bias,
                real_t *output, size_t batch_size, size_t input_dim,
                size_t output_dim);

/**
 * @brief Forward pass for 2D convolutional layer
 * @param input Input data array
 * @param weights Convolution kernel weights
 * @param bias Bias vector
 * @param output Output data array
 * @param batch_size Number of samples in the batch
 * @param input_channels Number of input channels
 * @param output_channels Number of output channels
 * @param kernel_h Height of the convolution kernel
 * @param kernel_w Width of the convolution kernel
 * @param h_in Height of input feature map
 * @param w_in Width of input feature map
 */
void conv2d_forward(const real_t *input, const real_t *weights,
                    const real_t *bias, real_t *output, size_t batch_size,
                    size_t input_channels, size_t output_channels,
                    size_t kernel_h, size_t kernel_w, size_t h_in, size_t w_in);

/**
 * @brief Forward pass for 2D max pooling layer
 * @param input Input data array
 * @param output Output data array
 * @param batch_size Number of samples in the batch
 * @param input_channels Number of input channels
 * @param output_channels Number of output channels
 * @param kernel_h Height of the pooling kernel
 * @param kernel_w Width of the pooling kernel
 * @param h_in Height of input feature map
 * @param w_in Width of input feature map
 */
void maxpool2d_forward(const real_t *input, real_t *output, size_t batch_size,
                       size_t input_channels, size_t output_channels,
                       size_t kernel_h, size_t kernel_w, size_t h_in,
                       size_t w_in);

/**
 * @brief Apply activation function to data
 * @param d Input data array
 * @param y Output data array
 * @param len Length of the data arrays
 * @param act Activation function type (0: Softmax, 1: ReLU, 2: Sigmoid)
 */
void apply_activation(real_t *d, real_t *y, int len, int act);

#endif // KERNELS_H