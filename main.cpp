#include <iostream>
#include <tuple>
#include "network.h"
#include "dataset_loader.h"

int main() {
	std::string model_cfg_file = "sample_models/model_cfg.json";
	std::string model_weights_file = "sample_models/mnist_model.bin";

	std::string dataset_file = "dataset/mnist_test.txt";

	DatasetLoader dataset_loader(dataset_file, 10000, 784, 1, true);

	real_t* data;
	real_t* target;
	std::tie(data, target) = dataset_loader.get_next_sample();

	real_t* data_device = nullptr;
	DatasetLoader::allocate_on_device(data, &data_device, 28*28);

	printf("Data: %f\n", data[0]);
	printf("Target: %f\n", target[0]);

	if (data_device != nullptr)
		printf("Allocated data2");

	Network model_mnist = Network::from_file(model_cfg_file, model_weights_file);

	//std::printf("Num layers: %zu", model_mnist.num_layers());
	//model_mnist.print_network_stats();

	real_t* output_device = model_mnist.forward(data_device);

	real_t* output = DatasetLoader::deallocate_from_device(&data_device, 10);

	//printf("Output: %f\n", output[0]);

	cudaFree(data_device);
	delete[] data;
	delete[] target;
	
	return 0;
}