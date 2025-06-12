#include "layer.h"
#include "kernels.h"
#include <stdexcept>

Layer::Layer()
    : type(NoneType), input_dim(0), output_dim(0), kernel_h(0), kernel_w(0),
      act(None), w(nullptr), b(nullptr), d_w(nullptr), d_b(nullptr) {}

Layer::Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
             Activation act_func) // Mainly linear
    : type(layer_type), input_dim(in_dim), output_dim(out_dim), act(act_func),
      w(nullptr), b(nullptr), d_w(nullptr), d_b(nullptr), kernel_h(0),
      kernel_w(0) {}

Layer::Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
             size_t kernel_height, size_t kernel_width,
             Activation act_func) // Conv2D
    : type(layer_type), input_dim(in_dim), output_dim(out_dim), act(act_func),
      w(nullptr), b(nullptr), d_w(nullptr), d_b(nullptr),
      kernel_h(kernel_height), kernel_w(kernel_width) {}

Layer::Layer(LayerType layer_type, size_t kernel_height,
             size_t kernel_width) // Flatten
    : type(layer_type), input_dim(0), output_dim(0), act(None), w(nullptr),
      b(nullptr), d_w(nullptr), d_b(nullptr), kernel_h(kernel_height),
      kernel_w(kernel_width) {}

Layer::Layer(LayerType layer_type) // Pooling
    : type(layer_type), input_dim(0), output_dim(0), act(None), w(nullptr),
      b(nullptr), d_w(nullptr), d_b(nullptr), kernel_h(0), kernel_w(0) {}

Layer::Layer(const Layer &other)
    : type(other.type), input_dim(other.input_dim),
      output_dim(other.output_dim), act(other.act), kernel_h(other.kernel_h),
      kernel_w(other.kernel_w), w(nullptr), b(nullptr), d_w(nullptr),
      d_b(nullptr) {

  if (other.w) {
    size_t w_size;
    if (type == Conv2D)
      w_size = input_dim * output_dim * other.kernel_h * other.kernel_w;
    else if (type == Linear)
      w_size = input_dim * output_dim;
    else
      w_size = 0;

    w = new real_t[w_size];
    std::memcpy(w, other.w, w_size * sizeof(real_t));
  }
  if (other.b) {
    b = new real_t[output_dim];
    std::memcpy(b, other.b, output_dim * sizeof(real_t));
  }

  if (other.d_w && other.d_b && w && b) {
    alloc_device();
  }
}

Layer &Layer::operator=(const Layer &other) {
  if (this == &other)
    return *this;

  delete[] w;
  delete[] b;
  if (d_w)
    cudaFree(d_w);
  if (d_b)
    cudaFree(d_b);

  type = other.type;
  input_dim = other.input_dim;
  output_dim = other.output_dim;
  kernel_h = other.kernel_h;
  kernel_w = other.kernel_w;
  act = other.act;

  w = nullptr;
  b = nullptr;
  d_w = nullptr;
  d_b = nullptr;

  size_t w_size;
  if (other.w) {
    if (type == Conv2D)
      w_size = input_dim * output_dim * other.kernel_h * other.kernel_w;
    else if (type == Linear)
      w_size = input_dim * output_dim;
    else
      w_size = 0;

    w = new real_t[w_size];
    std::memcpy(w, other.w, w_size * sizeof(real_t));
  }
  if (other.b) {
    b = new real_t[output_dim];
    std::memcpy(b, other.b, output_dim * sizeof(real_t));
  }

  if (other.d_w && other.d_b && w && b) {
    alloc_device();
  }

  return *this;
}

Layer::~Layer() {
  delete[] w;
  w = nullptr;
  delete[] b;
  b = nullptr;

  if (d_w)
    cudaFree(d_w);
  if (d_b)
    cudaFree(d_b);
}

void Layer::init_weights(real_t *w_init, real_t *b_init) {
  size_t w_size;
  if (type == Conv2D)
    w_size = input_dim * output_dim * kernel_h * kernel_w;
  else if (type == Linear)
    w_size = input_dim * output_dim;
  else
    w_size = 0;

  if (w == nullptr)
    w = new real_t[w_size];
  if (b == nullptr)
    b = new real_t[output_dim];

  std::memcpy(w, w_init, w_size * sizeof(real_t));
  std::memcpy(b, b_init, output_dim * sizeof(real_t));
}

size_t Layer::get_input_dim() const { return input_dim; }
size_t Layer::get_output_dim() const { return output_dim; };
size_t Layer::get_kernel_h() const { return kernel_h; };
size_t Layer::get_kernel_w() const { return kernel_w; };
LayerType Layer::get_type() const { return type; };

void Layer::alloc_device() {
  cudaError_t err;

  if (w == nullptr || b == nullptr) {
    throw std::runtime_error(
        "Host arrays not allocated. Call init_weights() first.");
  }

  size_t w_size;
  if (type == Conv2D)
    w_size = input_dim * output_dim * kernel_h * kernel_w;
  else if (type == Linear)
    w_size = input_dim * output_dim;
  else
    w_size = 0;

  err = cudaMalloc(&d_w, w_size * sizeof(real_t));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate device memory for weights: " +
                             std::string(cudaGetErrorString(err)));
  }
  err = cudaMemcpy(d_w, w, w_size * sizeof(real_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_w);
    throw std::runtime_error("Failed to copy weights to device: " +
                             std::string(cudaGetErrorString(err)));
  }

  err = cudaMalloc(&d_b, output_dim * sizeof(real_t));
  if (err != cudaSuccess) {
    cudaFree(d_w);
    throw std::runtime_error("Failed to allocate device memory for biases: " +
                             std::string(cudaGetErrorString(err)));
  }
  err = cudaMemcpy(d_b, b, output_dim * sizeof(real_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_w);
    cudaFree(d_b);
    throw std::runtime_error("Failed to copy biases to device: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void Layer::print_layer_stats() const {
  std::printf("Layer type: %d\n", type);
  std::printf("Layer input dimension: %zu\n", input_dim);
  std::printf("Layer output dimension: %zu\n", output_dim);
  std::printf("Activation function: %d\n", act);

  size_t weights_size;
  if (type == Conv2D)
    weights_size = input_dim * output_dim * kernel_h * kernel_w;
  else if (type == Linear)
    weights_size = input_dim * output_dim;
  else
    weights_size = 0;

  std::printf("Weights size: %zu bytes\n", weights_size * sizeof(real_t));

  std::printf("First 5 weights: ");
  for (size_t i = 0; i < std::min<size_t>(5, weights_size); ++i) {
    std::printf("%f ", w[i]);
  }
  std::printf("\n");

  std::printf("Biases size: %zu bytes\n", output_dim * sizeof(real_t));

  std::printf("First 5 biases: ");
  for (size_t i = 0; i < std::min<size_t>(5, output_dim); ++i) {
    std::printf("%f ", b[i]);
  }
  std::printf("\n");
}

void Layer::forward(const real_t *input, real_t *output) const {
  if (input == nullptr) {
    throw std::runtime_error("Input pointer is null.");
  }
  if (w == nullptr || b == nullptr) {
    throw std::runtime_error("Weights or biases not initialized.");
  }
  if (d_w == nullptr || d_b == nullptr) {
    throw std::runtime_error("Device weights or biases not allocated.");
  }

  /*printf("Forward pass through layer with input dimension %zu and output "
         "dimension %zu\n",
         input_dim, output_dim);*/

  real_t *fc_output = nullptr;
  cudaError_t err = cudaMalloc(&fc_output, output_dim * sizeof(real_t));
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate intermediate buffer: " +
                             std::string(cudaGetErrorString(err)));
  }

  fc_forward(input, d_w, d_b, fc_output, 1, input_dim, output_dim);
  cudaDeviceSynchronize();

  if (act != Activation::None) {
    apply_activation(fc_output, output, output_dim, act);
    cudaDeviceSynchronize();
  } else {
    err = cudaMemcpy(output, fc_output, output_dim * sizeof(real_t),
                     cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      cudaFree(fc_output);
      throw std::runtime_error("Failed to copy FC output: " +
                               std::string(cudaGetErrorString(err)));
    }
  }

  cudaFree(fc_output);
}