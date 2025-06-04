#ifndef KERNELS_H
#define KERNELS_H

#include "common.h"

void fc_forward(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    size_t batch_size, size_t input_dim, size_t output_dim);

void apply_activation(float* d, float* y, int len, int act);

#endif // KERNELS_H