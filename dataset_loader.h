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
	size_t data_dimension;
	size_t target_dimension;
	size_t dataset_size;
	size_t current_index;
	bool ignore_header;
	real_t* allocated_data;

public:
	DatasetLoader(const std::string& path, size_t size, 
		size_t data_dim, size_t target_dim, bool ignore_header);
	~DatasetLoader();
	std::pair<real_t*, real_t*> DatasetLoader::get_next_sample(bool normalize=true);
	static void allocate_on_device(real_t* data, real_t** allocated_data, size_t size);
	static real_t* deallocate_from_device(real_t** allocated_data, size_t data_size);
	
	size_t get_dataset_size() const;
};

#endif // DATASET_LOADER_H