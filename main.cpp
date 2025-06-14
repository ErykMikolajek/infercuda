#include "dataset_loader.h"
#include "layer.h"
#include "network.h"
#include <iostream>
#include <tuple>

int main() {
    // std::string model_cfg_file = "sample_models/mlp_model_cfg.json";
    // std::string model_weights_file = "sample_models/mlp_model.bin";

    std::string model_cfg_file = "sample_models/cnn_model_cfg.json";
    std::string model_weights_file = "sample_models/cnn_model.bin";

    std::string dataset_file = "dataset/mnist_test.txt";

    DatasetLoader dataset_loader(dataset_file, 10000, 784, 1, true);

    real_t *data;
    real_t *target;
    std::tie(data, target) = dataset_loader.get_next_sample();

    real_t *data_device = nullptr;
    DatasetLoader::allocate_on_device(data, &data_device, 28 * 28);

    printf("Target1: %f\n", target[0]);

    Network model_mnist =
        Network::from_file(model_cfg_file, model_weights_file);

    //model_mnist.print_network_stats();

    if (model_mnist.requires_input_dimensions()) {
        model_mnist.set_input_dim(28, 28); // Set dimensions for MNIST images
    }

    real_t *output_device = model_mnist.forward(data_device);

    size_t output_size = model_mnist.get_layer(model_mnist.num_layers() - 1).get_output_dim(); 
    real_t *output = DatasetLoader::deallocate_from_device(&output_device, output_size);

    for (int i = 0; i < output_size; ++i) {
    	printf("Number %d probability: %f\n", i, output[i]*100);
     }

    //// Another pass:
    std::tie(data, target) = dataset_loader.get_next_sample();

    DatasetLoader::allocate_on_device(data, &data_device, 28 * 28);

    printf("Target2: %f\n", target[0]);

    output_device = model_mnist.forward(data_device);
    output = DatasetLoader::deallocate_from_device(&output_device,
    output_size);

    for (int i = 0; i < output_size; ++i) {
    	printf("Number %d probability: %f\n", i, output[i] * 100);
    }

    delete[] data;
    delete[] target;
    delete[] output;

    return 0;
}