import os
import sys
from pathlib import Path

_, cfg, ckpt, path = sys.argv
print(cfg, ckpt, path)
onnx = (Path(path) / (Path(cfg).stem + '.onnx')).__str__()
os.system(f"python tools/deploy/export_onnx.py --cfg {cfg}  --ckpt  {ckpt} --onnx {path}")
os.system(f"python tools/deploy/export_trt.py --onnx {onnx}  --batch 1 --type FP32")
