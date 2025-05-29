#ifndef DEFINITIONS_H
#define DEFINITIONS_H

struct Layer { 
	int in, out; 
	bool bias;
	float *w, *b; 
	float *d_w, *d_b;
	int act;
};
struct Model { 
	char* name;
	Layer* layers;
	int n_layers; 
};

void load_model(struct Model*, const char*, const char*);
//void infer(Model*, float*, float*);

#endif // DEFINITIONS_H