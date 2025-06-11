#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
konwersja pytorch_model.bin → surowe wagi+biasy (.bin) + opis architektury (.json),
obsługuje conv2d, linear, activation, maxpool2d, flatten, softmax
użyj:
  python3 convert_cnn.py \
    --input pytorch_model.bin \
    --out_bin weights.bin \
    --out_cfg model.json \
    --in_channels 1 \
    --img_h 28 \
    --img_w 28 \
    --num_classes 10
"""

import argparse
import json
import torch
import numpy as np
import torch.nn as nn

# definicja twojego modelu
class CNN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, output_shape),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def main():
    p = argparse.ArgumentParser(description="konwersja pytorch_model.bin → bin + json (cnn)")
    p.add_argument("--input", "-i", required=True, help="ścieżka do pytorch_model.bin")
    p.add_argument("--out_bin", "-b", required=True, help="ścieżka wyjściowa dla binarnych wag")
    p.add_argument("--out_cfg", "-c", required=True, help="ścieżka wyjściowa dla pliku json")
    p.add_argument("--in_channels", type=int, required=True, help="liczba kanałów wejściowych (np. 1)")
    p.add_argument("--img_h", type=int, required=True, help="wysokość obrazu (np. 28)")
    p.add_argument("--img_w", type=int, required=True, help="szerokość obrazu (np. 28)")
    p.add_argument("--num_classes", type=int, required=True, help="liczba klas wyjściowych")
    args = p.parse_args()

    # 1) wczytaj state_dict
    state = torch.load(args.input, map_location="cpu")

    # 2) odtwórz model i załaduj wagi
    model = CNN((args.in_channels, args.img_h, args.img_w), args.num_classes)
    model.load_state_dict(state)

    # 3) iteruj po warstwach sekwencji
    cfg = {"model": {"architecture": {"layers": []}}}
    with open(args.out_bin, "wb") as f:
        for layer in model.model:
            if isinstance(layer, nn.Conv2d):
                name = f"model.{len(cfg['model']['architecture']['layers'])}"
                w = layer.weight.detach().cpu().numpy().astype("float32")
                b = layer.bias.detach().cpu().numpy().astype("float32")
                # opis
                cfg["model"]["architecture"]["layers"].append({
                    "type": "conv2d",
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": [layer.kernel_size[0], layer.kernel_size[1]],
                    "activation": "none"
                })
                # zapis wag
                f.write(w.flatten().tobytes())
                f.write(b.flatten().tobytes())

            elif isinstance(layer, nn.Linear):
                name = f"model.{len(cfg['model']['architecture']['layers'])}"
                w = layer.weight.detach().cpu().numpy().astype("float32")
                b = layer.bias.detach().cpu().numpy().astype("float32")
                cfg["model"]["architecture"]["layers"].append({
                    "type": "linear",
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "activation": "none"
                })
                f.write(w.flatten().tobytes())
                f.write(b.flatten().tobytes())

            elif isinstance(layer, nn.ReLU):
                cfg["model"]["architecture"]["layers"].append({
                    "type": "activation",
                    "activation": "relu"
                })

            elif isinstance(layer, nn.Softmax):
                cfg["model"]["architecture"]["layers"].append({
                    "type": "activation",
                    "activation": "softmax"
                })

            elif isinstance(layer, nn.MaxPool2d):
                cfg["model"]["architecture"]["layers"].append({
                    "type": "maxpool2d",
                    "kernel_size": [layer.kernel_size, layer.kernel_size]
                })

            elif isinstance(layer, nn.Flatten):
                cfg["model"]["architecture"]["layers"].append({
                    "type": "flatten"
                })

            else:
                raise RuntimeError(f"nieznany typ warstwy: {layer}")

    # 4) zapisz json
    with open(args.out_cfg, "w", encoding="utf-8") as jf:
        json.dump(cfg, jf, ensure_ascii=False, indent=2)

    print(f"gotowe: '{args.out_cfg}', '{args.out_bin}'")

if __name__ == "__main__":
    main()
