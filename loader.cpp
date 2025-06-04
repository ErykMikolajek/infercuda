#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "loader.h"
#include "layer.h"
#include "network.h"

using json = nlohmann::json;

std::pair<Layer*, size_t> Loader::load_cfg(const char* cfg) {
    std::ifstream file(cfg);
    if (!file.is_open()) {
        std::cerr << "Cannot open the file: " << cfg << std::endl;
        return std::make_pair(nullptr, 0);
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

        layers[i] = Layer(l["in_features"], l["out_features"], act);
    }

	return std::make_pair(layers, n_layers);
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
		//printf("Layer %d - number of weights: %d, number of biases: %d\n", i, in*out, out);
        size_t read_w = fread(weights, sizeof(real_t), in * out, file);

        /*for (int j = 0; j < in * out; j++) {
            if (weights[j] > 100 || weights[j] < -100) 
			printf("Layer %d: weight[%d] = %f\n", i, j, weights[j]);

            if (j == in*out -1)
				printf("Last weight of layer [%d] %f\n", j, weights[j]);
		}*/

        if (read_w != (size_t)(in * out)) {
            std::cerr << "Error reading weighs of layer: " << i << std::endl;
            fclose(file);
            return;
        }

        real_t* biases = new real_t[out];
        size_t read_b = fread(biases, sizeof(real_t), out, file);
        if (read_b != (size_t)out) {
            std::cerr << "Error reading biases of layer: " << i << std::endl;
            fclose(file);
            return;
        }

        /*for (int j = 0; j < out; j++) {
            if (biases[j] > 100 || biases[j] < -100)
            printf("Layer %d: bias[%d] = %f\n", i, j, biases[j]);

            if (j == out - 1)
				printf("Last bias of layer [%d]: %f\n", j, biases[j]);
        }*/
        
		layers[i].init_weights(weights, biases);
        delete[] weights;
		delete[] biases;
    }

    fclose(file);
}

void Loader::allocate_on_device(Network &n) {
    for (int i = 0; i < n.num_layers(); i++) {
		Layer& l = n.get_layer(i);
		l.alloc_device();
	}
}

Network& Loader::load_model(const std::string cfg, const std::string bin) {
    Layer* new_layers;
    size_t num_layers;
    std::tie(new_layers, num_layers) = load_cfg(cfg.c_str());

	load_weights(bin.c_str(), new_layers, num_layers);

	Network* net = new Network(new_layers, num_layers);
    allocate_on_device(*net);

	//delete[] new_layers;

    return *net;
}