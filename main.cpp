/**
 * @file main.cpp
 * @brief Main entry point for the neural network inference application
 *
 * This file contains the main function that demonstrates the usage of the
 * neural network inference system. It loads a pre-trained model, processes
 * MNIST test data, and performs inference on sample images.
 */

#include "dataset_loader.h"
#include "layer.h"
#include "network.h"
#include <iostream>
#include <tuple>

/**
 * @brief Main function that demonstrates neural network inference
 *
 * This function:
 * 1. Loads a pre-trained CNN model configuration and weights
 * 2. Initializes the MNIST test dataset loader
 * 3. Performs inference on two sample images
 * 4. Displays the prediction probabilities for each digit
 *
 * @return int Returns 0 on successful execution
 */
int main() {
    // std::string model_cfg_file = "sample_models/mlp_model_cfg.json";
    // std::string model_weights_file = "sample_models/mlp_model.bin";

    std::string model_cfg_file = "sample_models/cnn_model_cfg.json";
    std::string model_weights_file = "sample_models/cnn_model.bin";

    std::string dataset_file = "dataset/mnist_test.txt";

    // Initialize dataset loader for MNIST test set
    DatasetLoader dataset_loader(dataset_file, 10000, 784, 1, true);

    // Get the first sample from the dataset
    real_t *data;
    real_t *target;
    std::tie(data, target) = dataset_loader.get_next_sample();

    // Allocate memory on GPU for input data
    real_t *data_device = nullptr;
    DatasetLoader::allocate_on_device(data, &data_device, 28 * 28);

    printf("Target1: %f\n", target[0]);

    // Load the neural network model
    Network model_mnist =
        Network::from_file(model_cfg_file, model_weights_file);

    // model_mnist.print_network_stats();

    // Set input dimensions for MNIST images if required
    if (model_mnist.requires_input_dimensions()) {
        model_mnist.set_input_dim(28, 28); // Set dimensions for MNIST images
    }

    // Perform forward pass through the network
    real_t *output_device = model_mnist.forward(data_device);

    // Get output size and copy results back to host
    size_t output_size =
        model_mnist.get_layer(model_mnist.num_layers() - 1).get_output_dim();
    real_t *output =
        DatasetLoader::deallocate_from_device(&output_device, output_size);

    // Print probabilities for each digit
    for (int i = 0; i < output_size; ++i) {
        printf("Number %d probability: %f\n", i, output[i] * 100);
    }

    // Process second sample
    std::tie(data, target) = dataset_loader.get_next_sample();

    DatasetLoader::allocate_on_device(data, &data_device, 28 * 28);

    printf("Target2: %f\n", target[0]);

    output_device = model_mnist.forward(data_device);
    output = DatasetLoader::deallocate_from_device(&output_device, output_size);

    for (int i = 0; i < output_size; ++i) {
        printf("Number %d probability: %f\n", i, output[i] * 100);
    }

    // Clean up allocated memory
    delete[] data;
    delete[] target;
    delete[] output;

    return 0;
}