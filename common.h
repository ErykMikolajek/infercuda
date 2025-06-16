/**
 * @file common.h
 * @brief Common definitions and types used throughout the project
 *
 * This file contains common type definitions, constants, and includes
 * that are used across the neural network implementation.
 */

#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <limits>

using real_t = float; ///< Type definition for real numbers (using float)

constexpr int BLOCK_SIZE =
    256; ///< Default CUDA block size for kernel execution
constexpr real_t REAL_MAX =
    std::numeric_limits<float>::max(); ///< Maximum value for real_t type

#endif // COMMON_H