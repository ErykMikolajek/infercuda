#ifndef NETWORK_H
#define NETWORK_H

#include "common.h"
#include "layer.h"
#include <vector>
#include <string>
//#include "loader.h"

class Network
{
    private:
        Layer* layers;
		size_t n_layers;
		
	public:
		Network();
		Network(Layer* layers);
		~Network();
        static Network from_file(const std::string& filepath);
		size_t num_layers() const;
		Layer& get_layer(size_t layer_index) const;

};

#endif // NETWORK_H