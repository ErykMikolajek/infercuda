#include "layer.h"

Layer::Layer(size_t in_dim, size_t out_dim, Activation act_func) : input_dim(in_dim), output_dim(out_dim), act(act_func) {
	w = b = d_w = d_b = nullptr;
}

Layer::Layer() {}

Layer::~Layer() {
	
}

void Layer::init_weights(real_t* w_init, real_t* b_init) {
	this->w = w_init;
	this->b = b_init;
}

size_t Layer::get_input_dim() const { return input_dim; }
size_t Layer::get_output_dim() const { return output_dim; };

void Layer::alloc_device() {
    cudaMalloc(&d_w, input_dim * output_dim * sizeof(real_t));
    cudaMemcpy(d_w, w, input_dim * output_dim * sizeof(real_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_b, output_dim * sizeof(real_t));
    cudaMemcpy(&d_b, b, output_dim * sizeof(real_t), cudaMemcpyHostToDevice);
}