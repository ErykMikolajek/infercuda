#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

#include "common.h"
#include <string>
#include <fstream>

class DatasetLoader
{
private:
	std::string dataset_path;
	std::ifstream dataset_file;
	size_t dataset_size;
	size_t current_index;

public:
	DatasetLoader(const std::string& path, size_t size, size_t data_dim, size_t target_dim);
	~DatasetLoader();
	void load_next_sample();
	
	size_t getDatasetSize() const;
};

#endif // DATASET_LOADER_H