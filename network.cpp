#include "network.h"
#include "layer.h"
#include "loader.h"


Network::Network() {
	layers = nullptr;
	n_layers = 0;
}

Network::Network(Layer* layers): layers(layers) {
	n_layers = sizeof(layers) / sizeof(Layer);
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