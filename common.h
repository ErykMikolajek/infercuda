#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using real_t = float;
constexpr int BLOCK_SIZE = 256;
constexpr real_t FLT_MAX = std::numeric_limits<float>::max();

#endif // COMMON_H