struct Layer { 
	int in, out; 
	bool bias;
	float* w, *b; 
	int act;
};
struct Model { 
	Layer* layers;
	int n_layers; 
};

void load_model(Model*, const char*, const char*);
void infer(Model*, float*, float*);