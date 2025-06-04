#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
skrypt konwertuje huggingface’owy pytorch_model.bin → surowe wagi+biasy (.bin) + opis architektury (.json).
użyj: python3 convert_hf_mnist.py --input pytorch_model.bin --out_bin weights.bin --out_cfg model.json
"""

import argparse
import json
import torch


def main():
    p = argparse.ArgumentParser(description="konwersja huggingface pytorch_model.bin → bin + json")
    p.add_argument("--input", "-i", required=True, help="ścieżka do pytorch_model.bin")
    p.add_argument("--out_bin", "-b", required=True, help="ścieżka wyjściowa dla surowych wag+biasów")
    p.add_argument("--out_cfg", "-c", required=True, help="ścieżka wyjściowa dla jsona z architekturą")
    args = p.parse_args()

    # 1) wczytaj state_dict (pickle)
    state = torch.load(args.input, map_location="cpu")

    # 2) wydziel pary (layer_name → {weight, bias})
    layers = {}
    for k, v in state.items():
        # spodziewamy się kluczy typu "<nazwa_warstwy>.weight" lub ".bias"
        if k.endswith(".weight"):
            name = k[: -len(".weight")]
            layers.setdefault(name, {})["weight"] = v.cpu().numpy().astype("float32")
        elif k.endswith(".bias"):
            name = k[: -len(".bias")]
            layers.setdefault(name, {})["bias"] = v.cpu().numpy().astype("float32")

    # sortuj warstwy wg nazwy (lexicograficznie)
    sorted_names = sorted(layers.keys())

    # 3) zbuduj opis architektury
    arch = {"model": {"architecture": {"layers": []}}}
    for name in sorted_names:
        d = layers[name]
        w = d.get("weight")
        b = d.get("bias")
        if w is None or b is None:
            raise RuntimeError(f"brakuje weight lub bias dla warstwy '{name}'")
        # w.shape == (out_features, in_features)
        out_f, in_f = w.shape
        arch["model"]["architecture"]["layers"].append({
            "in_features": int(in_f),
            "out_features": int(out_f),
            "activation": "none"  # ustaw na odpowiednią, jeśli wiesz jaka
        })

    # zapisz json
    with open(args.out_cfg, "w", encoding="utf-8") as f:
        json.dump(arch, f, ensure_ascii=False, indent=2)

    # 4) wypakuj wagi+biasy w binarną sekwencję float32 (waga warstwa po warstwie, potem bias)
    with open(args.out_bin, "wb") as f:
        for name in sorted_names:
            w = layers[name]["weight"]
            b = layers[name]["bias"]
            # flatten C-order: w[0][0..], w[1][..] itd.
            f.write(w.tobytes())
            f.write(b.tobytes())

    print(f"gotowe: '{args.out_cfg}' + '{args.out_bin}'")


if __name__ == "__main__":
    main()
