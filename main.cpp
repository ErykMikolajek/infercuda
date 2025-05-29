#include <iostream>
#include "network.h"

int main() {
	std::string model_cfg_file = "sample_models/model_cfg.json";
	std::string model_weights_file = "sample_models/mnist_model.bin";

	Network model_mnist = Network::from_file(model_cfg_file, model_weights_file);
}