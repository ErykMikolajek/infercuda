#include <iostream>
#include "loader.cpp"
#include "model_definitions.h"

int main() {
	char* model_cfg_file = "sample_models/model_cfg.json";
	char* model_weights_file = "sample_models/mnist_model.bin";

	struct Model new_inference_model;
	new_inference_model.name = "mnist_model";
	load_model(&new_inference_model, model_cfg_file, model_weights_file);
}