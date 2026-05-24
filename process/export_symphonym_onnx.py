#!/usr/bin/env python3
"""Export the Symphonym v7 Student (UniversalEncoder) to ONNX for in-browser phonetic name search.

The reference forward (hf/inference.py) uses pack_padded_sequence, which doesn't export to ONNX — but
for a SINGLE unpadded query the packing and the attention mask are no-ops, so we wrap the trained
sub-modules in a packing-free, mask-free forward (identical output for batch=1, no padding) and export
that. Inputs: char_ids (1,L) + script_id (1,) + lang_id (1,) + length (1,). Output: (1,128) L2-normalised.

  python3 process/export_symphonym_onnx.py --sym /home/stephen/PycharmProjects/indexing/hf \
      --out docs/search/symphonym.onnx
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class ExportEncoder(torch.nn.Module):
    """Packing-free / mask-free wrapper around a trained UniversalEncoder (valid for batch=1, no pad)."""
    def __init__(self, enc):
        super().__init__()
        self.e = enc

    def forward(self, char_ids, script_id, lang_id, length):
        e = self.e
        L = char_ids.shape[1]
        c = e.char_embed(char_ids)
        s = e.script_embed(script_id).unsqueeze(1).expand(-1, L, -1)
        l = e.lang_embed(lang_id).unsqueeze(1).expand(-1, L, -1)
        lb = ((length - 1) // 2).clamp(0, e.num_length_buckets - 1)
        le = e.length_embed(lb).unsqueeze(1).expand(-1, L, -1)
        x = torch.cat([c, s, l, le], dim=-1)
        x = e.input_norm(e.input_proj(x))
        out, _ = e.bilstm(x)                 # full sequence, no packing
        att, _ = e.self_attention(out, None) # no mask (all positions valid)
        att = att + out
        pooled, _ = e.pooling(att, None)
        return F.normalize(e.output_proj(pooled), p=2, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="/home/stephen/PycharmProjects/indexing/hf", help="Symphonym hf/ dir")
    ap.add_argument("--out", default="docs/search/symphonym.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    sys.path.insert(0, args.sym)
    from inference import SymphonymModel        # noqa: E402

    sm = SymphonymModel(model_dir=args.sym)
    wrap = ExportEncoder(sm._model).eval()

    # sample input ("London")
    ci, si, li, ln = sm._tokenise("London", "und")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrap, (ci, si, li, ln), args.out, opset_version=args.opset, dynamo=False,  # legacy tracer handles dynamic seq
        input_names=["char_ids", "script_id", "lang_id", "length"], output_names=["embedding"],
        dynamic_axes={"char_ids": {1: "seq"}})
    print(f"exported -> {args.out} ({Path(args.out).stat().st_size/1e6:.1f} MB)")

    # numerical parity: ONNX vs the reference SymphonymModel.embed
    import onnxruntime as ort
    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    print(f"\n{'name':<16}{'cos(onnx, torch)':>18}")
    for name in ["London", "Москва", "Köln", "Lussac-les-Églises", "Constantinople"]:
        ci, si, li, ln = sm._tokenise(name, "und")
        out = sess.run(None, {"char_ids": ci.numpy(), "script_id": si.numpy(),
                              "lang_id": li.numpy(), "length": ln.numpy()})[0][0]
        ref = sm.embed(name, "und")
        print(f"{name:<16}{float(np.dot(out, ref)):>18.6f}")


if __name__ == "__main__":
    main()
