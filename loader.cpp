#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "loader.h"
#include "layer.h"
#include "network.h"

using json = nlohmann::json;

Layer* Loader::load_cfg(const char* cfg) {
    std::ifstream file(cfg);
    if (!file.is_open()) {
        std::cerr << "Cannot open the file: " << cfg << std::endl;
        return nullptr;
    }
    json j;
    try
    {
        j = json::parse(file);
    }
    catch (json::parse_error& ex)
    {
        std::cerr << "Config file parse error at byte " << ex.byte << std::endl;
    }

    auto layers_json = j["model"]["architecture"]["layers"];
    int n_layers = layers_json.size();
	Layer* layers = new Layer[n_layers];

    for (int i = 0; i < n_layers; ++i) {
        const auto& l = layers_json[i];

        std::string act_string = l["activation"];
        Activation act;
        
        if (act_string == "relu") act = Activation::ReLU;
        else if (act_string == "softmax") act = Activation::Softmax;
		else act = Activation::None;

        new (&layers[i]) Layer(l["in_features"], l["out_features"], act);       
    }

	return layers;
}

void Loader::load_weights(const char* bin, Layer* layers, size_t n_layers) {
    FILE* file = fopen(bin, "rb");
    if (!file) {
        std::cerr << "Error opening file: " << bin << std::endl;
        return;
    }

    for (int i = 0; i < n_layers; i++) {
		int in = layers[i].get_input_dim();
        int out = layers[i].get_output_dim();
     
        real_t* weights = new real_t[in * out];
        size_t read_w = fread(weights, sizeof(float), in * out, file);
        if (read_w != (size_t)(in * out)) {
            std::cerr << "Error reading weighs of layer: " << i << std::endl;
            fclose(file);
            return;
        }

        real_t* biases = new real_t[out];
        size_t read_b = fread(biases, sizeof(float), out, file);
        if (read_b != (size_t)out) {
            std::cerr << "Error reading biases of layer: " << i << std::endl;
            fclose(file);
            return;
        }
        
		layers[i].init_weights(weights, biases);
        delete[] weights;
		delete[] biases;
    }

    fclose(file);
}

void Loader::allocate_on_device(Network &n) {
    for (int i = 0; i < n.num_layers(); i++) {
		Layer l = n.get_layer(i);
		l.alloc_device();
	}
}

Network& Loader::load_model(const std::string cfg, const std::string bin) {
	Layer* new_layers = load_cfg(cfg.c_str());
	size_t n_layers = sizeof(new_layers) / sizeof(new_layers[0]);
	load_weights(bin.c_str(), new_layers, n_layers);

	Network net = Network(new_layers);
    allocate_on_device(net);

    return net;
}