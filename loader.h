#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>

class Network;
class Layer;

class Loader {
  public:
    static Network &load_model(const std::string cfg, const std::string bin);

  private:
    static void load_weights(const char *bin, Layer *layers, size_t n_layers);
    static std::pair<Layer *, size_t> load_cfg(const char *cfg);
    static void allocate_on_device(Network &m);
};

#endif