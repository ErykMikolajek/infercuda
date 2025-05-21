# infercuda
Minimal neural network inference engine written in pure C with CUDA acceleration. Supports basic feedforward nets with relu/sigmoid and dense layers. Designed for learning and experimentation with gpu computation

### Project structure
```
include/
└── model_definitions.cuh       ← structures definitions + prototypes
src/
├── main.c               ← arg parsing, init, infer → teardown
├── loader.c             ← load_bin, load_cfg, allocs host/device
├── infer.c              ← infer_host (calls next layer_infer)
└── utils.c              ← cpu_gemv, cpu_activation, benchmarks
kernels/
├── fc_layer.cu          ← gemv_kernel + bias + activation
├── conv_layer.cu        ← direct_conv_kernel + activation
├── pool_layer.cu        ← maxpool_kernel + avgpool_kernel
└── activations.cu       ← relu_kernel, softmax_kernel
```