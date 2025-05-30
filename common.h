#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <memory>

using real_t = float;
constexpr int BLOCK_SIZE = 256;

#endif // COMMON_H