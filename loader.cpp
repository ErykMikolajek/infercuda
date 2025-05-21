#include "include/model_definitions.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void load_model(Model* m, const char* cfg, const char* bin) {
    load_cfg(cfg, m);
    load_weights(bin, m->layers, m->n_layers);
	//alloc_device(m);
}

void load_weights(const char* bin, Layer* layers, int n_layers) {
    FILE* file = fopen(bin, "rb");
    if (!file) {
        std::cerr << "Error opening file: " << bin << std::endl;
        return;
    }

    for (int i = 0; i < n_layers; i++){
        int in = layers[i].in;
        int out = layers[i].out;
        bool has_bias = layers[i].bias;

        layers[i].w = new float[in * out];
        size_t read_w = fread(layers[i].w, sizeof(float), in * out, file);
        if (read_w != (size_t)(in * out)) {
            std::cerr << "Error reading weighs of layer: " << i << std::endl;
            fclose(file);
            return;
        }

        if (has_bias) {
            layers[i].b = new float[out];
            size_t read_b = fread(layers[i].b, sizeof(float), out, file);
            if (read_b != (size_t)out) {
                std::cerr << "Error reading biases of layer: " << i << std::endl;
                fclose(file);
                return;
            }
        } else {
            layers[i].b = nullptr;
        }
    }

    fclose(file);
}

void load_cfg(const char* cfg, Model* m) {
    std::ifstream file(cfg);
    if (!file.is_open()) {
        std::cerr << "Cannot open the file: " << cfg << std::endl;
        return;
    }

    json j;
    file >> j;

    auto layers_json = j["model"]["architecture"]["layers"];
    int n_layers = layers_json.size();
    m->n_layers = n_layers;
    m->layers = new Layer[n_layers];

    for (int i = 0; i < n_layers; ++i) {
        const auto& l = layers_json[i];
        m->layers[i].in = l["in_features"];
        m->layers[i].out = l["out_features"];
		m->layers[i].bias = l["bias"];
        m->layers[i].w = nullptr;
        m->layers[i].b = nullptr;

        std::string act = l["activation"];
        if (act == "relu") m->layers[i].act = 1;
        else if (act == "softmax") m->layers[i].act = 2;
        else m->layers[i].act = 0;
    }
}

//void alloc_device(Model* m);