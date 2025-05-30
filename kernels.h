#ifndef KERNELS_H
#define KERNELS_H

#include "common.h"

void fc_forward(const real_t* input, const real_t* weights,
    const real_t* bias, real_t* output,
    size_t batch_size, size_t input_dim, size_t output_dim);

void relu_forward(const real_t* input, real_t* output);

void sigmoid_forward(const real_t* input, real_t* output);

#endif