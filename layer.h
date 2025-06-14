#ifndef LAYER_H
#define LAYER_H

#include "common.h"

enum Activation { None = -1, Softmax = 0, ReLU = 1, Sigmoid = 2 };

enum LayerType {
    NoneType = -1,
    Linear = 0,
    Conv2D = 1,
    Flatten = 2,
    Pooling = 3
};

struct LayerDimensions {
    size_t height;
    size_t width;
    size_t channels;
};

class Layer {
  private:
    size_t input_dim, output_dim;
    size_t kernel_h, kernel_w;

    real_t *d_w, *d_b;
    real_t *w, *b;

    Activation act;
    LayerType type;

  public:
    Layer();
    Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
          Activation act_func);
    Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
          size_t kernel_height, size_t kernel_width, Activation act_func);
    Layer::Layer(LayerType layer_type, size_t kernel_height,
        size_t kernel_width, size_t in_dim);
    Layer(LayerType layer_type);
    Layer(const Layer &other);
    Layer &operator=(const Layer &other);
    ~Layer();

    void init_weights(real_t *w_init, real_t *b_init);
    size_t get_input_dim() const;
    size_t get_output_dim() const;
    size_t get_kernel_h() const;
    size_t get_kernel_w() const;
    LayerType get_type() const;
    LayerDimensions get_output_dimensions(size_t input_height,
                                          size_t input_width,
                                          size_t input_channels) const;
    void alloc_device();

    void forward(const real_t *input, real_t *output, size_t h_in = 0,
                 size_t w_in = 0) const;

    void print_layer_stats() const;
};

#endif // LAYER_H