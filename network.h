#ifndef NETWORK_H
#define NETWORK_H

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
		Network(Layer* layers);
		~Network();
        static Network& from_file(const std::string config_file, const std::string binary_file);
		size_t num_layers() const;
		Layer& get_layer(size_t layer_index) const;

};

#endif // NETWORK_H