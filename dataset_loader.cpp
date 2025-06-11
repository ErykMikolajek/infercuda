#include "dataset_loader.h"

DatasetLoader::DatasetLoader(const std::string& path, size_t size, 
	size_t data_dim, size_t target_dim, bool ignore_file_header)
	: dataset_path(path), dataset_size(size), current_index(0), 
	  data_dimension(data_dim), target_dimension(target_dim), ignore_header(ignore_file_header) {
	
    dataset_file = std::ifstream(dataset_path, std::ios::binary);
	if (!dataset_file.is_open()) {
		throw std::runtime_error("Could not open dataset file: " + dataset_path);
	}
}

DatasetLoader::~DatasetLoader() {
	dataset_file.close();
}


size_t DatasetLoader::get_dataset_size() const {
	return dataset_size;
}


std::pair<real_t*, real_t*> DatasetLoader::get_next_sample(bool normalize) {
    if (current_index >= dataset_size) {
        return std::make_pair(nullptr, nullptr);
    }
    if (ignore_header && current_index == 0) {
        std::string header_line;
        std::getline(dataset_file, header_line);
	}

    real_t* target = new real_t[target_dimension];
    real_t* data = new real_t[data_dimension];

    for (size_t i = 0; i < target_dimension; ++i) {
        int temp;
        if (!(dataset_file >> temp)) {
            delete[] target;
            delete[] data;
            return std::make_pair(nullptr, nullptr);
        }
		//printf("Target[%zu]: %d\n", i, temp); // Debug output
        target[i] = static_cast<real_t>(temp);
    }

    for (size_t i = 0; i < data_dimension; ++i) {
        int temp;
        if (!(dataset_file >> temp)) {
            delete[] target;
            delete[] data;
            return std::make_pair(nullptr, nullptr);
        }
		//printf("Data[%zu]: %d\n", i, temp); // Debug output
        data[i] = static_cast<real_t>(temp);
    }

    if (normalize) {
        real_t data_min = data[0], data_max = data[0];
        for (size_t i = 1; i < data_dimension; ++i) {
            if (data[i] < data_min) data_min = data[i];
            if (data[i] > data_max) data_max = data[i];
        }

        if (data_max != data_min) {
            for (size_t i = 0; i < data_dimension; ++i) {
                data[i] = (data[i] - data_min) / (data_max - data_min);
            }
        }
    }

    current_index++;
    return std::make_pair(data, target);
}



void DatasetLoader::allocate_on_device(real_t* data, real_t** allocated_data, size_t size) {
    cudaError_t err;

    if (data == nullptr) {
        throw std::runtime_error("Source data is empty");
    }

    err = cudaMalloc(allocated_data, size * sizeof(real_t));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for data: " +
            std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpy(*allocated_data, data, size * sizeof(real_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*allocated_data);
        throw std::runtime_error("Failed to copy weights to device: " +
            std::string(cudaGetErrorString(err)));
    }
    /*if (allocated_data != nullptr)
        printf("Data allocated successfully\n");*/
}

real_t* DatasetLoader::deallocate_from_device(real_t** allocated_data, size_t data_size) {
    cudaError_t err;

    if (allocated_data == nullptr || *allocated_data == nullptr) {
        throw std::runtime_error("Allocated data pointer is null");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize device: " +
            std::string(cudaGetErrorString(err)));
    }

    real_t* data = new real_t[data_size];

    err = cudaMemcpy(data, *allocated_data, data_size * sizeof(real_t),
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        delete[] data;
        throw std::runtime_error("Failed to copy data from device: " +
            std::string(cudaGetErrorString(err)));
    }

    err = cudaFree(*allocated_data);
    if (err != cudaSuccess) {
        delete[] data;
        throw std::runtime_error("Failed to free device memory: " +
            std::string(cudaGetErrorString(err)));
    }

    *allocated_data = nullptr;
    return data;
}