struct Layer { int in, out, k_h, k_w, s; float* w, * b; char act; };
struct Model { Layer* layers; int n_layers; };

void load_model(Model*, char*, char*);
void infer(Model*, float*, float*);