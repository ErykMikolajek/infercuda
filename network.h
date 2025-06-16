/**
 * @file network.h
 * @brief Header file defining the Network class for neural network management
 *
 * This file contains the Network class definition which manages a collection of
 * neural network layers and provides functionality for loading models from
 * files and performing inference.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"
#include <cstdio>
#include <string>
#include <vector>

class Layer;

/**
 * @class Network
 * @brief Class representing a complete neural network
 *
 * This class manages a collection of neural network layers and provides
 * functionality for model loading, inference, and network statistics.
 */
class Network {
  private:
    Layer *layers;   ///< Array of network layers
    size_t n_layers; ///< Number of layers in the network
    size_t h_in = 0; ///< Input height (for 2D inputs)
    size_t w_in = 0; ///< Input width (for 2D inputs)

  public:
    /**
     * @brief Default constructor
     */
    Network();

    /**
     * @brief Constructor with pre-defined layers
     * @param layers Array of layers
     * @param num_layers Number of layers
     */
    Network(Layer *layers, size_t num_layers);

    /**
     * @brief Copy constructor
     * @param other Network to copy from
     */
    Network(const Network &other);

    /**
     * @brief Assignment operator
     * @param other Network to assign from
     * @return Reference to this network
     */
    Network &operator=(const Network &other);

    /**
     * @brief Destructor
     */
    ~Network();

    /**
     * @brief Create a network from configuration and weight files
     * @param config_file Path to the network configuration file
     * @param binary_file Path to the network weights file
     * @return Reference to the created network
     */
    static Network &from_file(const std::string config_file,
                              const std::string binary_file);

    /**
     * @brief Get the number of layers in the network
     * @return size_t Number of layers
     */
    size_t num_layers() const;

    /**
     * @brief Set input dimensions for 2D inputs
     * @param h Input height
     * @param w Input width
     */
    void set_input_dim(size_t h, size_t w);

    /**
     * @brief Get a reference to a specific layer
     * @param layer_index Index of the layer to get
     * @return Reference to the requested layer
     */
    Layer &get_layer(size_t layer_index) const;

    /**
     * @brief Check if the network requires input dimensions to be set
     * @return bool True if input dimensions are required
     */
    bool requires_input_dimensions() const;

    /**
     * @brief Perform forward propagation through the network
     * @param input Input data
     * @return Pointer to the output data
     */
    real_t *forward(real_t *input) const;

    /**
     * @brief Print network statistics
     */
    void print_network_stats() const;
};

#endif // NETWORK_H