/**
 * @file dataset_loader.h
 * @brief Header file for the DatasetLoader class that handles loading and
 * preprocessing of datasets
 *
 * This file defines the DatasetLoader class which is responsible for:
 * - Loading dataset files
 * - Managing dataset samples
 * - Handling GPU memory allocation for data
 * - Providing access to dataset samples
 */

#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "common.h"
#include <fstream>
#include <string>

/**
 * @class DatasetLoader
 * @brief Class for loading and managing dataset samples
 *
 * This class provides functionality to load dataset files, manage samples,
 * and handle GPU memory allocation for data processing.
 */
class DatasetLoader {
  private:
    std::string dataset_path;   ///< Path to the dataset file
    std::ifstream dataset_file; ///< File stream for reading the dataset
    size_t data_dimension;      ///< Dimension of input data
    size_t target_dimension;    ///< Dimension of target data
    size_t dataset_size;        ///< Total number of samples in the dataset
    size_t current_index;       ///< Current position in the dataset
    bool ignore_header;     ///< Whether to ignore the first line of the dataset
    real_t *allocated_data; ///< Pointer to allocated data memory

  public:
    /**
     * @brief Constructor for DatasetLoader
     * @param path Path to the dataset file
     * @param size Total number of samples in the dataset
     * @param data_dim Dimension of input data
     * @param target_dim Dimension of target data
     * @param ignore_header Whether to ignore the first line of the dataset
     */
    DatasetLoader(const std::string &path, size_t size, size_t data_dim,
                  size_t target_dim, bool ignore_header);

    /**
     * @brief Destructor for DatasetLoader
     * Cleans up allocated resources
     */
    ~DatasetLoader();

    /**
     * @brief Get the next sample from the dataset
     * @param normalize Whether to normalize the data (default: true)
     * @return std::pair containing pointers to data and target arrays
     */
    std::pair<real_t *, real_t *> get_next_sample(bool normalize = true);

    /**
     * @brief Allocate memory on the GPU device
     * @param data Pointer to the source data
     * @param allocated_data Pointer to store the allocated device memory
     * @param size Size of the data to allocate
     */
    static void allocate_on_device(real_t *data, real_t **allocated_data,
                                   size_t size);

    /**
     * @brief Deallocate memory from the GPU device and copy data back to host
     * @param allocated_data Pointer to the device memory to deallocate
     * @param data_size Size of the data
     * @return Pointer to the host memory containing the data
     */
    static real_t *deallocate_from_device(real_t **allocated_data,
                                          size_t data_size);

    /**
     * @brief Get the total size of the dataset
     * @return size_t Total number of samples in the dataset
     */
    size_t get_dataset_size() const;
};

#endif // DATASET_LOADER_H