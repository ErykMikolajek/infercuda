#include "network.h"
#include "layer.h"
#include "loader.h"


Network::Network() {
	layers = nullptr;
	n_layers = 0;
}

Network::Network(Layer* external_layers, size_t num_layers): n_layers(num_layers) {
    layers = new Layer[n_layers];
    for (size_t i = 0; i < n_layers; ++i) {
        layers[i] = external_layers[i];
    }
}

Network::Network(const Network& other) : n_layers(other.n_layers) {
    if (n_layers > 0) {
        layers = new Layer[n_layers];
        for (size_t i = 0; i < n_layers; ++i) {
            layers[i] = other.layers[i];
        }
    } else {
        layers = nullptr;
    }
}

Network& Network::operator=(const Network& other) {
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

Network& Network::from_file(const std::string config_file, const std::string binary_file) {
	return Loader::load_model(config_file, binary_file);
}

size_t Network::num_layers() const {
	return n_layers;
}

Layer& Network::get_layer(size_t layer_index) const {
	return layers[layer_index];
}

void Network::print_network_stats() const {
	std::printf("---- Network stats: ----");
	for (size_t i = 0; i < n_layers; ++i) {
		std::printf("\n------- Layer %zu: -------\n", i);
		layers[i].print_layer_stats();
	}
}