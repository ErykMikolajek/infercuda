#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "common.h"
#include "layer.h"
#include "network.h"
#include <string>

class Loader {
public:
    static Network load_model(const std::string cfg, const std::string bin);

private:
    static void load_weights(const char* bin, Layer* layers, int n_layers);
    static Layer* load_cfg(const char* cfg);
    static void allocate_on_device(Network &m);
};

#endif