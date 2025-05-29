#ifndef LAYER_H
#define LAYER_H

#include "common.h"

enum Activation{
	None = -1,
	Softmax = 0,
	ReLU = 1,
	Sigmoid = 2
};

class Layer {
private:
	size_t input_dim, output_dim;
    real_t* d_w, * d_b;
    real_t* w, * b;
	Activation act;

public:
	Layer();
	Layer(size_t in_dim, size_t out_dim, Activation act_func);
    ~Layer();

	void init_weights(real_t* w_init, real_t* b_init);
	size_t get_input_dim() const;
	size_t get_output_dim() const;
	void alloc_device();
};

#endif