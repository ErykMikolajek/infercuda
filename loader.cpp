#include "loader.h"
#include "layer.h"
#include "network.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sys/types.h>

using json = nlohmann::json;

std::pair<Layer *, size_t> Loader::load_cfg(const char *cfg) {
  std::ifstream file(cfg);
  if (!file.is_open()) {
    std::cerr << "Cannot open the file: " << cfg << std::endl;
    return std::make_pair(nullptr, 0);
  }
  json j;
  try {
    j = json::parse(file);
  } catch (json::parse_error &ex) {
    std::cerr << "Config file parse error at byte " << ex.byte << std::endl;
  }

  auto layers_json = j["model"]["architecture"]["layers"];
  int n_layers = layers_json.size();
  Layer *layers = new Layer[n_layers];

  for (int i = 0; i < n_layers; ++i) {
    const auto &l = layers_json[i];

    std::string act_string = l["activation"];
    Activation act;

    if (act_string == "relu")
      act = Activation::ReLU;
    else if (act_string == "softmax")
      act = Activation::Softmax;
    else
      act = Activation::None;

    switch (l["type"]) {
    case "conv2d":
      layers[i] = Layer(LayerType::Conv2D, l["in_features"], l["out_features"],
                        l["kernel_size"][0], l["kernel_size"][1], act);
      break;
    case "linear":
      layers[i] =
          Layer(LayerType::Linear, l["in_features"], l["out_features"], act);
      break;
    case "maxpool2d":
      layers[i] =
          Layer(LayerType::Pooling, l["kernel_size"][0], l["kernel_size"][1]);
      break;
    case "flatten":
      layers[i] = Layer(LayerType::Flatten);
      break;
    default:
      layers[i] = Layer();
      break;
    }
  }

  return std::make_pair(layers, n_layers);
}

void Loader::load_weights(const char *bin, Layer *layers, size_t n_layers) {
  FILE *file = fopen(bin, "rb");
  if (!file) {
    std::cerr << "Error opening file: " << bin << std::endl;
    return;
  }

  for (int i = 0; i < n_layers; i++) {
    int in_dim = layers[i].get_input_dim();
    int out_dim = layers[i].get_output_dim();

    size_t w_size;
    switch (layers[i].get_type()) {
    case LayerType::Conv2D:
      w_size = in_dim * out_dim * layers[i].get_kernel_h() *
               layers[i].get_kernel_w();
      break;
    case LayerType::Linear:
      w_size = in_dim * out_dim;
      break;
    case LayerType::Pooling:
    case LayerType::Flatten:
    default:
      w_size = 0;
      break;
    }

    if (w_size > 0) {
      real_t *weights = new real_t[w_size];
      size_t read_w = fread(weights, sizeof(real_t), w_size, file);

      //// DEBUG PRINT
      for (int j = 0; j < w_size; j++) {
        if (weights[j] > 100 || weights[j] < -100)
          printf("Layer %d: weight[%d] = %f\n", i, j, weights[j]);

        if (j == w_size - 1)
          printf("Last weight of layer [%d] %f\n", j, weights[j]);
      }
      //// DEBUG PRINT

      if (read_w != (size_t)w_size) {
        std::cerr << "Error reading weights of layer: " << i << std::endl;
        fclose(file);
        return;
      }
      real_t *biases = new real_t[out_dim];
      size_t read_b = fread(biases, sizeof(real_t), out_dim, file);

      //// DEBUG PRINT
      for (int j = 0; j < out_dim; j++) {
        if (biases[j] > 100 || biases[j] < -100)
          printf("Layer %d: bias[%d] = %f\n", i, j, biases[j]);

        if (j == out_dim - 1)
          printf("Last bias of layer [%d]: %f\n", j, biases[j]);
      }
      //// DEBUG PRINT

      if (read_b != (size_t)out_dim) {
        std::cerr << "Error reading biases of layer: " << i << std::endl;
        fclose(file);
        return;
      }
      layers[i].init_weights(weights, biases);

      delete[] weights;
      delete[] biases;
    }
  }

  fclose(file);
}

void Loader::allocate_on_device(Network &n) {
  for (int i = 0; i < n.num_layers(); i++) {
    Layer &l = n.get_layer(i);
    l.alloc_device();
  }
}

Network &Loader::load_model(const std::string cfg, const std::string bin) {
  Layer *new_layers;
  size_t num_layers;
  std::tie(new_layers, num_layers) = load_cfg(cfg.c_str());

  load_weights(bin.c_str(), new_layers, num_layers);

  Network *net = new Network(new_layers, num_layers);
  allocate_on_device(*net);

  // delete[] new_layers;

  return *net;
}