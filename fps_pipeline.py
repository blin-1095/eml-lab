import torch
import numpy as np

from torch_detection_pipeline import TorchDetectionPipeline
from onnx_detection_pipeline import OnnxDetectionPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_models = [
    ("Torch baseline", "state_dicts/voc_pretrained.pt", 20),
]
"""
List of all torch models to test. 
Format: (name, path, num_classes)
"""

onnx_models = [
    ("ONNX baseline", "models/baseline_best_model.onnx"),
    ("ONNX person", "models/person_best_model.onnx"),
    ("ONNX augmented", "models/augmented_best_model.onnx"),
    ("ONNX pruned 0", "models/pruned_pipeline_0_best_model.onnx"),
    ("ONNX pruned 20", "models/pruned_pipeline_20_best_model.onnx"),
    ("ONNX pruned 40", "models/pruned_pipeline_40_best_model.onnx"),
    ("ONNX pruned 60", "models/pruned_pipeline_60_best_model.onnx"),
    ("ONNX pruned 80", "models/pruned_pipeline_80_best_model.onnx"),
]
"""
List of all onnx models to test. 
Format: (name, path)
"""

results = {}

# Torch models
for name, path, num_classes in torch_models:
    pipeline = TorchDetectionPipeline(num_classes, path, device)
    results[name] = pipeline.fps_pipeline()

    del pipeline

    if device.type == "cuda":
        torch.cuda.empty_cache()

# ONNX models
for name, path in onnx_models:
    pipeline = OnnxDetectionPipeline(path, device)
    results[name] = pipeline.fps_pipeline()

    del pipeline

    if device.type == "cuda":
        torch.cuda.empty_cache()

import matplotlib.pyplot as plt

names = list(results.keys())
fps = np.round(list(results.values()))

plt.figure(figsize=(10, 6))
bars = plt.barh(names, fps, color="steelblue")

plt.title("Average FPS of Detection Pipelines")
plt.xlabel("Average FPS")

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.1f}",
        va="center"
    )

# Highest FPS at the top
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("results/fps_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()