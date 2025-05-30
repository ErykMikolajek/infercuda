#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"
#include <cstdio>
#include <vector>
#include <string>

class Layer;

class Network
{
    private:
        Layer* layers;
		size_t n_layers;
		
	public:
		Network();
		Network(Layer* layers, size_t num_layers);
		Network(const Network& other);
		Network& operator=(const Network& other);
		~Network();
        static Network& from_file(const std::string config_file, const std::string binary_file);
		size_t num_layers() const;
		Layer& get_layer(size_t layer_index) const;

		real_t* forward(real_t* input) const;

		void print_network_stats() const;

};

#endif // NETWORK_H