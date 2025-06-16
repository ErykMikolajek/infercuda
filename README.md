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
-   Configurable network architecture through JSON configuration
-   Optimized CUDA kernels for parallel processing

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
-   Implementing efficient memory access patterns for GPU operations

## 3. Implementation

**Structure**  
The project is organized into several key components:

#### Core Classes:

-   Layer: Base class for neural network layers
-   Network: Manages the neural network architecture
-   DatasetLoader: Handles dataset loading and preprocessing
-   Loader: Manages neural net model loading from files

#### Layer Types:

1. **Linear Layer**

    - Fully connected layer for dense neural networks
    - Implements matrix multiplication with weights and bias addition
    - Supports batch processing through CUDA kernels

2. **Conv2D Layer**

    - 2D Convolutional layer for image processing
    - Supports multiple input and output channels
    - Implements efficient CUDA kernels with shared memory tiling
    - Configurable kernel size and padding

3. **Pooling Layer**

    - Max pooling operation for dimensionality reduction
    - Supports configurable kernel size
    - Efficient parallel implementation using CUDA

4. **Flatten Layer**
    - Reshapes multi-dimensional input into 1D vector
    - Used for transitioning between convolutional and dense layers

#### Activation Functions:

1. **ReLU (Rectified Linear Unit)**

    - Implementation: f(x) = max(0, x)
    - Efficient parallel computation on GPU
    - Helps with vanishing gradient problem

2. **Softmax**
    - Implementation: f(x_i) = exp(x_i - max) / sum(exp(x_j - max))
    - Includes numerical stability optimization
    - Used for multi-class classification output

#### CUDA Kernels:

1. **Fully Connected Layer Kernels**

    - Optimized matrix multiplication
    - Support for both single and batch processing
    - Efficient memory access patterns

2. **Convolutional Layer Kernels**

    - Shared memory tiling for improved performance
    - Parallel processing of multiple channels
    - Optimized memory access for input and weight tensors

3. **Pooling Layer Kernels**

    - Parallel reduction for finding maximum values
    - Efficient handling of overlapping regions
    - Optimized memory access patterns

4. **Activation Function Kernels**
    - Parallel element-wise operations
    - Efficient implementation of ReLU and Softmax
    - Numerical stability optimizations for Softmax

**File Structure:**

```
infercuda/
├── include/
│   ├── common.h            # Common definitions and types
│   ├── layer.h             # Layer class definition
│   ├── network.h           # Network class definition
│   ├── loader.h            # Model loading utilities
│   ├── kernels.h           # CUDA kernel declarations
│   └── dataset_loader.h    # Dataset handling utilities
├── src/
│   ├── layer.cpp           # Layer class implementation
│   ├── network.cpp         # Network class implementation
│   ├── loader.cpp          # Model loading implementation
│   └── dataset_loader.cpp  # Dataset handling implementation
├── cuda/
│   ├── activations.cu      # Activation function kernels
│   ├── conv_kernel.cu      # Convolutional layer kernels
│   ├── fc_kernel.cu        # Fully connected layer kernels
│   └── maxpool_kernel.cu   # Pooling layer kernels
├── sample_models/          # Example model configurations
├── dataset/                # Dataset storage
```

**Environment**

-   C++ compiler with CUDA support
-   NVIDIA GPU with CUDA capability
-   Required dependencies:
    -   CUDA Toolkit
    -   C++ Standard Library
    -   Standard I/O libraries
    -   nlohmann/json for model configuration parsing

## 4. Performance Comparison

The following performance comparisons were conducted on the same machine to ensure fair benchmarking. The tests were performed using both MLP (Multi-Layer Perceptron) and CNN (Convolutional Neural Network) architectures.

### Hardware Configuration

-   CPU: AMD Ryzen 5 2600
-   GPU: GTX 1060 3GB
-   RAM: 8GB
-   CUDA Version: 12.9

### MLP Model Performance

| Framework | Device | Average Inference Time (s) | Speedup vs CPU |
| --------- | ------ | -------------------------- | -------------- |
| PyTorch   | CPU    | 9.62                       | 1x (baseline)  |
| PyTorch   | GPU    | 11.53                      | 1.19x          |
| InferCUDA | GPU    | 19.97                      | 2.08x          |

### Key Observations

**MLP Performance**

    - InferCUDA shows 2.08x degradation over PyTorch CPU
    - Compared to PyTorch GPU, InferCUDA is 1.7x slower
    - The InferCUDA project is very simple and not properly opitimized yet, so the numbers are worse than the state-of-the art machine learning frameworks
    - The dataset loader might also be the cause for higher inference times as pytroch uses preloaded datasets

### Methodology

-   All tests were performed using the same input data
-   Average of 1000 inference runs was taken
-   Batch size of 1 was used for all tests
-   No data preprocessing time was included in measurements

## 5. Testing

**The project includes several testing aspects:**

Integration Tests:

-   End-to-end inference: manual data flow through the model
-   Model loading: debug print functions to validate loaded model
-   Dataset processing
-   Memory management validation
-   Layer type compatibility checks

Performance Tests:

-   Inference speed measurements
-   Memory transfer efficiency
-   GPU utilization metrics
-   Kernel execution time profiling

## 6. Future Improvements

A number of improvements are needed to expand this project capabilities to fully functional CUDA inference C++ library:

-   **Performance Optimizations:**
    -   Optimize some of the CUDA kernels - due to time and complexity constraints not all of the kernel could be fully implemented with performance metrics in mind
    -   Add support for multi batch processing - for the time being only single data samples can be processed at once by the network
    -   Optimize memory transfers - data loading from the dataset could be further improved to maximize speed
-   **Feature Additions:**
    -   Support for more layer types - the current implementation involve most basic layer types such as Linear and Conv2D
    -   Additional activation functions - functions such as Sigmoid, GELU, Leaky ReLU or others could be implemented to further increase project capabilities
    -   Batch normalization - this mechanism would also be needed to support more advanced models
    -   Support for recurrent layers (LSTM, GRU)
    -   Attention mechanisms for transformer architectures
-   **Usability Enhancements:**
    -   Better error handling - although thorough error handling has been already implemented if this project would be aimed for public usage a more general and elegant error handling system would need to be implemented
    -   Support for more model formats - loader class could be enhanced with the ability to support more model and layer types
    -   Add comprehensive logging system
    -   Implement configuration validation
    -   Add model visualization tools
-   **Development Tools:**
    -   Performance monitoring - more advanced performance measuring system could be implemented to ease the process of testing
    -   Add unit testing framework
    -   Implement continuous integration
    -   Add benchmarking suite
    -   Create development documentation

## 7. User Manual

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

## 8. How to run

**Prerequisites**

1. Install CUDA Toolkit (version 11.0 or higher recommended)
2. Install C++ compiler with CUDA support (GCC 9.0+ or Clang 10.0+)
3. Set up build environment
4. Install nlohmann/json library for JSON parsing

#### Running the Example

1. Clone the repository:

```bash
git clone https://github.com/yourusername/infercuda.git
cd infercuda
```

2. Build the project:

```bash
mkdir build && cd build
cmake ..
make
```

3. Prepare the model and data:

-   Place model configuration in `sample_models`: `sample_models/cnn_model_cfg.json`
-   Place model weights in `sample_models`: `sample_models/cnn_model.bin`
-   Place your dataset in `dataset/mnist_test.txt`

4. Run the example:

```bash
./infercuda_example
```

**Expected Output**

1. CUDA Errors:
    - Check GPU compatibility
    - Verify CUDA installation
    - Check memory availability
    - Ensure proper CUDA driver installation
2. Build Errors:
    - Verify CUDA toolkit installation
    - Check compiler compatibility
    - Ensure all dependencies are installed
    - Check CMake configuration
3. Runtime Errors:
    - Check model file paths
    - Verify dataset format
    - Monitor GPU memory usage
    - Check input dimensions
    - Validate model configuration

Please ensure your code follows the project's coding style and includes appropriate documentation.

## 9. License

This project is licensed under the MIT License - see the LICENSE file for details.
