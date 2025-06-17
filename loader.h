/**
 * @file loader.h
 * @brief Model loading functionality for neural networks
 *
 * This file defines the Loader class which is responsible for loading
 * neural network models from configuration and weight files.
 */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>

class Network;
class Layer;

/**
 * @class Loader
 * @brief Class for loading neural network models from files
 *
 * This class provides static methods for loading neural network models
 * from configuration and weight files, and handling device memory allocation.
 */
class Loader {
  public:
    /**
     * @brief Load a neural network model from files
     * @param cfg Path to the model configuration file
     * @param bin Path to the model weights file
     * @return Reference to the loaded network
     */
    static Network &load_model(const std::string cfg, const std::string bin);

  private:
    /**
     * @brief Load network weights from binary file
     * @param bin Path to the weights file
     * @param layers Array of layers to load weights into
     * @param n_layers Number of layers
     */
    static void load_weights(const char *bin, Layer *layers, size_t n_layers);

    /**
     * @brief Load network configuration from file
     * @param cfg Path to the configuration file
     * @return Pair containing array of layers and number of layers
     */
    static std::pair<Layer *, size_t> load_cfg(const char *cfg);

    /**
     * @brief Allocate memory on device for network
     * @param m Network to allocate memory for
     */
    static void allocate_on_device(Network &m);
};

#endif