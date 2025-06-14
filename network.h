#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"
#include <cstdio>
#include <string>
#include <vector>

class Layer;

class Network {
  private:
    Layer *layers;
    size_t n_layers;
    size_t h_in = 0;
    size_t w_in = 0;

  public:
    Network();
    Network(Layer *layers, size_t num_layers);
    Network(const Network &other);
    Network &operator=(const Network &other);
    ~Network();
    static Network &from_file(const std::string config_file,
                              const std::string binary_file);
    size_t num_layers() const;
    void set_input_dim(size_t h, size_t w);
    Layer &get_layer(size_t layer_index) const;
    bool requires_input_dimensions() const;

    real_t *forward(real_t *input) const;

    void print_network_stats() const;
};

#endif // NETWORK_H