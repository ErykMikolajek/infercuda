# InferCUDA: Neural Network Inference Engine

## 1. Project Description

InferCUDA is a minimal neural network inference engine written in C++ with CUDA acceleration. It is designed to provide efficient GPU-accelerated inference for neural networks, with a focus on learning and experimentation with GPU computation. The project supports various types of neural network layers and operations, making it suitable for both educational purposes and practical applications.

**Key features:**

-   CUDA-accelerated neural network inference
-   Support for multiple layer types (Linear, Conv2D, Flatten, Pooling)
-   Multiple activation functions (Softmax, ReLU, Sigmoid)
-   Efficient memory management for GPU operations
-   Neural net loading from configuration and binary files
-   Dataset loading and preprocessing capabilities
-   Support for MNIST dataset format

## 2. Problem and Solution

**Problem**  
Neural network inference, especially for complex models, can be computationally intensive when performed on CPUs.

-   This leads to:
-   Slow inference times
-   Limited throughput for batch processing
-   Inefficient resource utilization
-   High latency in real-time applications

**Solution**  
InferCUDA addresses these challenges by:

-   Utilizing NVIDIA GPUs through CUDA for parallel processing
-   Implementing optimized CUDA kernels for different neural network operations
-   Providing efficient memory management between CPU and GPU
-   Supporting various neural network architectures
-   Offering a simple yet powerful API for model loading and inference

## 3. Implementation

**Structure**  
The project is organized into several key components:

#### Core Classes:

-   Layer: Base class for neural network layers
-   Network: Manages the neural network architecture
-   DatasetLoader: Handles dataset loading and preprocessing
-   Loader: Manages neural net model loading from files

#### CUDA Kernels:

-   Fully connected layer operations
-   Convolutional layer operations
-   Pooling operations
-   Activation functions

**File Structure:**

```
include/
└── model_definitions.cuh       ← structures definitions + prototypes
src/
├── main.c               ← arg parsing, init, infer → teardown
├── loader.c             ← load_bin, load_cfg, allocs host/device
├── infer.c              ← infer_host (calls next layer_infer)
└── utils.c              ← cpu_gemv, cpu_activation, benchmarks
kernels/
├── fc_layer.cu          ← gemv_kernel + bias + activation
├── conv_layer.cu        ← direct_conv_kernel + activation
├── pool_layer.cu        ← maxpool_kernel + avgpool_kernel
└── activations.cu       ← relu_kernel, softmax_kernel
```

**Environment**

-   C++ compiler with CUDA support
-   NVIDIA GPU with CUDA capability
-   Required dependencies:
    -   CUDA Toolkit
    -   C++ Standard Library
    -   Standard I/O libraries

## 4. Testing

**The project includes several testing aspects:**

Integration Tests:

-   End-to-end inference: manual data flow through the model
-   Model loading: debug print functions to validate loaded model
-   Dataset processing

Performance Tests:

-   Inference speed

## 5. Future Improvements

A number of improvements are needed to expand this project capabilities to fully functional CUDA inference C++ library:

-   **Performance Optimizations:**
    -   Optimize some of the CUDA kernels - due to time and complexity constraints not all of the kernel could be fully implemented with performance metrics in mind
    -   Add support for multi batch processing - for the time being only single data samples can be processed at once by the network
    -   Optimize memory transfers - data loading from the dataset could be further improved to maximize speed
-   **Feature Additions:**
    -   Support for more layer types - the current implementation involve most basic layer types such as Linear and Conv2D
    -   Additional activation functions - functions such as Sigmoid, GELU, Leaky ReLU or others could be implemented to further increase project capabilities
    -   Batch normalization - this mechanism would also be needed to support more advanced models
-   **Usability Enhancements:**
    -   Better error handling - although thorough error handling has been already implemented if this project would be aimed for public usage a more general and elegant error handling system would need to be implemented
    -   Support for more model formats - loader class could be enhanced with the ability to support more model and layer types
-   **Development Tools:**
    -   Performance monitoring - more advanced performance measuring system could be implemented to ease the process of testing

## 6. User Manual

**Basic Usage**

1. Loading a model:

```cpp
Network model = Network::from_file("model_cfg.json", "model_weights.bin");
```

2. Setting Input Dimensions: (only needed when loading CNN net)

```cpp
model.set_input_dim(height, width);
```

3. Performing Inference:

```cpp
real_t* output = model.forward(input_data);
```

**Dataset Loading:**

1. Initialize DatasetLoader:

```cpp
DatasetLoader loader("dataset.txt", size, data_dim, target_dim, ignore_header);
```

2. Get Samples:

```cpp
auto [data, target] = loader.get_next_sample();
```

**Memory Management:**

1. GPU Memory Allocation:

```cpp
real_t* device_data = nullptr;
DatasetLoader::allocate_on_device(data, &device_data, size);
```

2. GPU Memory Deallocation:

```cpp
real_t* host_data = DatasetLoader::deallocate_from_device(&device_data, size);
```

## 7. How to run

**Prerequisites**

1. Install CUDA Toolkit
2. Install C++ compiler with CUDA support
3. Set up build environment

#### Running the Example

-   Place model configuration in `sample_models`: `sample_models/cnn_model_cfg.json`
-   Place model weights in `sample_models`: `sample_models/cnn_model.bin`
-   Place your dataset in `dataset/mnist_test.txt`

**Expected Output**

1. CUDA Errors:
    - Check GPU compatibility
    - Verify CUDA installation
    - Check memory availability
2. Build Errors:
    - Verify CUDA toolkit installation
    - Check compiler compatibility
    - Ensure all dependencies are installed
3. Runtime Errors:
    - Check model file paths
    - Verify dataset format
    - Monitor GPU memory usage

##

This documentation provides a comprehensive overview of the InferCUDA project, its implementation, and usage. For more detailed information about specific components or features, refer to the inline documentation in the source code.
