/**
 * @file layer.h
 * @brief Header file defining neural network layer types and the Layer class
 *
 * This file contains definitions for different types of neural network layers,
 * activation functions, and the Layer class that implements the core
 * functionality of neural network layers.
 */

#ifndef LAYER_H
#define LAYER_H

#include "common.h"

/**
 * @enum Activation
 * @brief Enumeration of supported activation functions
 */
enum Activation { None = -1, Softmax = 0, ReLU = 1, Sigmoid = 2 };

/**
 * @enum LayerType
 * @brief Enumeration of supported layer types
 */
enum LayerType {
    NoneType = -1,
    Linear = 0,
    Conv2D = 1,
    Flatten = 2,
    Pooling = 3
};

/**
 * @struct LayerDimensions
 * @brief Structure holding dimensions for 2D layers
 */
struct LayerDimensions {
    size_t height;   ///< Height of the layer output
    size_t width;    ///< Width of the layer output
    size_t channels; ///< Number of channels in the layer output
};

/**
 * @class Layer
 * @brief Base class for neural network layers
 *
 * This class implements the core functionality of neural network layers,
 * including weight management, forward propagation, and dimension handling.
 */
class Layer {
  private:
    size_t input_dim, output_dim; ///< Input and output dimensions
    size_t kernel_h, kernel_w; ///< Kernel dimensions for convolutional layers

    real_t *d_w, *d_b; ///< Device pointers for weights and biases
    real_t *w, *b;     ///< Host pointers for weights and biases

    Activation act; ///< Activation function type
    LayerType type; ///< Layer type

  public:
    /**
     * @brief Default constructor
     */
    Layer();

    /**
     * @brief Constructor for basic layers
     * @param layer_type Type of the layer
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     * @param act_func Activation function
     */
    Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
          Activation act_func);

    /**
     * @brief Constructor for convolutional layers
     * @param layer_type Type of the layer
     * @param in_dim Input dimension
     * @param out_dim Output dimension
     * @param kernel_height Height of the convolution kernel
     * @param kernel_width Width of the convolution kernel
     * @param act_func Activation function
     */
    Layer(LayerType layer_type, size_t in_dim, size_t out_dim,
          size_t kernel_height, size_t kernel_width, Activation act_func);

    /**
     * @brief Constructor for pooling layers
     * @param layer_type Type of the layer
     * @param kernel_height Height of the pooling kernel
     * @param kernel_width Width of the pooling kernel
     * @param in_dim Input dimension
     */
    Layer(LayerType layer_type, size_t kernel_height, size_t kernel_width,
          size_t in_dim);

    /**
     * @brief Constructor for flatten layers
     * @param layer_type Type of the layer
     * @param out_dim Output dimension
     */
    Layer(LayerType layer_type, size_t out_dim);

    /**
     * @brief Copy constructor
     * @param other Layer to copy from
     */
    Layer(const Layer &other);

    /**
     * @brief Assignment operator
     * @param other Layer to assign from
     * @return Reference to this layer
     */
    Layer &operator=(const Layer &other);

    /**
     * @brief Destructor
     */
    ~Layer();

    /**
     * @brief Initialize layer weights
     * @param w_init Initial weights
     * @param b_init Initial biases
     */
    void init_weights(real_t *w_init, real_t *b_init);

    /**
     * @brief Get input dimension
     * @return size_t Input dimension
     */
    size_t get_input_dim() const;

    /**
     * @brief Get output dimension
     * @return size_t Output dimension
     */
    size_t get_output_dim() const;

    /**
     * @brief Get kernel height
     * @return size_t Kernel height
     */
    size_t get_kernel_h() const;

    /**
     * @brief Get kernel width
     * @return size_t Kernel width
     */
    size_t get_kernel_w() const;

    /**
     * @brief Get layer type
     * @return LayerType Type of the layer
     */
    LayerType get_type() const;

    /**
     * @brief Calculate output dimensions for 2D layers
     * @param input_height Height of input
     * @param input_width Width of input
     * @param input_channels Number of input channels
     * @return LayerDimensions Output dimensions
     */
    LayerDimensions get_output_dimensions(size_t input_height,
                                          size_t input_width,
                                          size_t input_channels) const;

    /**
     * @brief Allocate memory on device
     */
    void alloc_device();

    /**
     * @brief Perform forward propagation
     * @param input Input data
     * @param output Output data
     * @param h_in Input height (for 2D layers)
     * @param w_in Input width (for 2D layers)
     */
    void forward(const real_t *input, real_t *output, size_t h_in = 0,
                 size_t w_in = 0) const;

    /**
     * @brief Print layer statistics
     */
    void print_layer_stats() const;
};

#endif // LAYER_H