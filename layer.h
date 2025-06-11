#ifndef LAYER_H
#define LAYER_H

#include "common.h"
//#include "kernels.h"

enum Activation{
	None = -1,
	Softmax = 0,
	ReLU = 1,
	Sigmoid = 2
};

enum LayerType {
	None = -1,
	Linear = 0,
	Conv2D = 1,
	Flatten = 2,
	Pooling = 3
};

class Layer {
private:
	size_t input_dim, output_dim;
	size_t kernel_h, kernel_w;

    real_t* d_w, * d_b;
    real_t* w, * b;

	Activation act;
	LayerType type;

public:
	Layer();
	Layer(size_t in_dim, size_t out_dim, Activation act_func);
	Layer::Layer(LayerType layer_type, size_t in_dim, size_t out_dim, Activation act_func);
	Layer::Layer(LayerType layer_type, size_t in_dim, size_t out_dim, size_t kernel_height, size_t kernel_width, Activation act_func);
	Layer::Layer(LayerType layer_type, size_t kernel_height, size_t kernel_width);
	Layer(const Layer& other);
	Layer& operator=(const Layer& other);
    ~Layer();

	void init_weights(real_t* w_init, real_t* b_init);
	size_t get_input_dim() const;
	size_t get_output_dim() const;
	void alloc_device();

	void forward(const real_t* input, real_t* output) const;

	void print_layer_stats() const;
};

#endif // LAYER_H