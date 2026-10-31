#!/usr/bin/env python3
"""Evaluate pretrained vs fine-tuned YOLOv8n on test set with density stratification."""
import os
from pathlib import Path
import pandas as pd
import numpy as np
from ultralytics import YOLO
import json

ROOT = Path("/workspace/artifacts/yolox_retail_shelf/real_shelf_experiment")
TEST_IMG = ROOT / "dataset" / "images" / "test"
TEST_LBL = ROOT / "dataset" / "labels" / "test"
DENSITY_CSV = ROOT / "results" / "image_density.csv"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

# Load density info
density_df = pd.read_csv(DENSITY_CSV)
test_density = density_df[density_df["split"] == "test"].set_index("image")

# Load GT counts from labels
gt_counts = {}
for lbl in TEST_LBL.glob("*.txt"):
    name = lbl.stem
    with open(lbl) as f:
        n = sum(1 for line in f if line.strip())
    gt_counts[name] = n

print("Test GT counts:", gt_counts)

# Models
pretrained = YOLO("yolov8n.pt")  # COCO pretrained
finetuned_path = ROOT / "runs" / "detect" / "runs" / "finetuned_yolov8n" / "weights" / "best.pt"
finetuned = YOLO(str(finetuned_path))

def count_predictions(model, img_path, conf=0.25):
    results = model.predict(str(img_path), conf=conf, imgsz=640, device="cpu", verbose=False, max_det=1000)
    return len(results[0].boxes) if results and results[0].boxes is not None else 0

# Evaluate both
rows = []
for img_path in sorted(TEST_IMG.glob("*.jpg")):
    name = img_path.stem
    gt = gt_counts.get(name, 0)
    dens_info = test_density.loc[name] if name in test_density.index else None
    
    pred_pre = count_predictions(pretrained, img_path)
    pred_ft = count_predictions(finetuned, img_path)
    
    rows.append({
        "image": name,
        "gt_count": gt,
        "pred_pretrained": pred_pre,
        "pred_finetuned": pred_ft,
        "error_pretrained": pred_pre - gt,
        "error_finetuned": pred_ft - gt,
        "abs_error_pretrained": abs(pred_pre - gt),
        "abs_error_finetuned": abs(pred_ft - gt),
    })
    print(f"{name}: GT={gt}, Pre={pred_pre}, FT={pred_ft}")

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "per_image_results.csv", index=False)

# Density tiers on test set (low/med/high by tertiles)
counts = df["gt_count"].values
q33, q66 = np.percentile(counts, [33, 66])
def tier(c):
    if c <= q33:
        return "low"
    elif c <= q66:
        return "medium"
    else:
        return "high"
df["density_tier"] = df["gt_count"].apply(tier)
print(f"\nDensity thresholds: low<={q33:.0f}, med<={q66:.0f}")

# Summary by density
summary = df.groupby("density_tier").agg({
    "gt_count": "mean",
    "abs_error_pretrained": "mean",
    "abs_error_finetuned": "mean",
    "error_pretrained": "mean",
    "error_finetuned": "mean",
}).round(1)
summary["mae_reduction_pct"] = ((summary["abs_error_pretrained"] - summary["abs_error_finetuned"]) / summary["abs_error_pretrained"] * 100).round(1)
print("\nSummary by density tier:")
print(summary)
summary.to_csv(OUT_DIR / "summary_by_density.csv")

# Overall
print("\nOverall MAE:")
print(f"  Pretrained: {df['abs_error_pretrained'].mean():.1f}")
print(f"  Fine-tuned: {df['abs_error_finetuned'].mean():.1f}")
print(f"  Reduction: {(1 - df['abs_error_finetuned'].mean() / df['abs_error_pretrained'].mean()) * 100:.1f}%")

print(f"\nResults saved to {OUT_DIR}")
