#ifndef KERNELS_H
#define KERNELS_H

#include "common.h"

void fc_forward(const real_t *input, const real_t *weights, const real_t *bias,
                real_t *output, size_t batch_size, size_t input_dim,
                size_t output_dim);

void conv2d_forward(const real_t *input, const real_t *weights,
                    const real_t *bias, real_t *output, size_t batch_size,
                    size_t input_channels, size_t output_channels,
                    size_t kernel_h, size_t kernel_w, size_t h_in, size_t w_in);

void maxpool2d_forward(const real_t *input, real_t *output, size_t batch_size,
                       size_t input_channels, size_t output_channels, size_t kernel_h,
                       size_t kernel_w, size_t h_in, size_t w_in);

void apply_activation(real_t *d, real_t *y, int len, int act);

#endif // KERNELS_H