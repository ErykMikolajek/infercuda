#include "dataset_loader.h"

DatasetLoader::DatasetLoader(const std::string& path, size_t size, size_t ) 
	: dataset_path(path), dataset_size(size), current_index(0) {
	
    dataset_file = std::ifstream(dataset_path, std::ios::binary);
	if (!dataset_file.is_open()) {
		throw std::runtime_error("Could not open dataset file: " + dataset_path);
	}

}

DatasetLoader::~DatasetLoader() {
	dataset_file.close();
}

std::pair<real_t, real_t> DatasetLoader::load_next_sample(bool normalize) {

}

size_t getDatasetSize() const;