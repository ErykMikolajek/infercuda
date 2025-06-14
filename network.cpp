#include "network.h"
#include "layer.h"
#include "loader.h"
#include <stdexcept>

Network::Network() {
    layers = nullptr;
    n_layers = 0;
}

Network::Network(Layer *external_layers, size_t num_layers)
    : n_layers(num_layers) {
    layers = new Layer[n_layers];
    for (size_t i = 0; i < n_layers; ++i) {
        layers[i] = external_layers[i];
    }
}

Network::Network(const Network &other) : n_layers(other.n_layers) {
    if (n_layers > 0) {
        layers = new Layer[n_layers];
        for (size_t i = 0; i < n_layers; ++i) {
            layers[i] = other.layers[i];
        }
    } else {
        layers = nullptr;
    }
}

Network &Network::operator=(const Network &other) {
    if (this == &other) {
        return *this;
    }

    if (layers != nullptr) {
        delete[] layers;
        layers = nullptr;
    }

    n_layers = other.n_layers;

    if (n_layers > 0) {
        layers = new Layer[n_layers];
        for (size_t i = 0; i < n_layers; ++i) {
            layers[i] = other.layers[i];
        }
    } else {
        layers = nullptr;
    }

    return *this;
}

Network::~Network() {
    if (layers != nullptr) {
        delete[] layers;
        layers = nullptr;
    }
    n_layers = 0;
}

Network &Network::from_file(const std::string config_file,
                            const std::string binary_file) {
    return Loader::load_model(config_file, binary_file);
}

size_t Network::num_layers() const { return n_layers; }

Layer &Network::get_layer(size_t layer_index) const {
    return layers[layer_index];
}

void Network::set_input_dim(size_t h, size_t w) {
    h_in = h;
    w_in = w;
}

void Network::print_network_stats() const {
    std::printf("---- Network stats: ----");
    for (size_t i = 0; i < n_layers; ++i) {
        std::printf("\n------- Layer %zu: -------\n", i);
        layers[i].print_layer_stats();
    }
}

bool Network::requires_input_dimensions() const {
    if (n_layers == 0)
        return false;
    return layers[0].get_type() == LayerType::Conv2D;
}

real_t *Network::forward(real_t *input) const {
    if (requires_input_dimensions() && (h_in == 0 || w_in == 0)) {
        throw std::runtime_error("Input dimensions (height and width) must be "
                                 "set for Conv2D networks. "
                                 "Call set_input_dim() before forward()");
    }

    if (n_layers == 0) {
        return input;
    }

    size_t final_output_size = layers[n_layers - 1].get_output_dim();
    real_t *final_output = nullptr;
    cudaError_t err =
        cudaMalloc(&final_output, final_output_size * sizeof(real_t));

    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate final output buffer: " +
                                 std::string(cudaGetErrorString(err)));
    }

    real_t *current_input = input;
    real_t *current_output = nullptr;

    size_t h_in = this->h_in;
    size_t w_in = this->w_in;

    for (size_t i = 0; i < n_layers; ++i) {

        auto output_dims = layers[i].get_output_dimensions(
            h_in, w_in, layers[i].get_input_dim());
        size_t output_size =
            output_dims.height * output_dims.width * output_dims.channels;

        if (i == n_layers - 1) {
            current_output = final_output;
        } else {
            err = cudaMalloc(&current_output, output_size * sizeof(real_t));
            if (err != cudaSuccess) {
                if (i > 0)
                    cudaFree(current_input);
                cudaFree(final_output);
                throw std::runtime_error(
                    "Failed to allocate intermediate output buffer: " +
                    std::string(cudaGetErrorString(err)));
            }
        }

        if (layers[i].get_type() == LayerType::Conv2D) {
            layers[i].forward(current_input, current_output, h_in, w_in);
        } else {
            layers[i].forward(current_input, current_output);
        }

        if (i > 0 && current_input != input) {
            cudaFree(current_input);
        }

        current_input = current_output;
        h_in = output_dims.height;
        w_in = output_dims.width;
    }

    cudaDeviceSynchronize();

    return final_output;
}