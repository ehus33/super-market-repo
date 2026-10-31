# Supermarket Shelf Product Detection

Two-model experiment comparing a COCO-pretrained detector against a fine-tuned model on the Humans in the Loop Supermarket Shelves dataset.

## Experiment Summary

| Metric | Pretrained (COCO) | Fine-tuned | Improvement |
|--------|-------------------|------------|-------------|
| mAP@0.5 | 0.005 | **0.442** | ~88× |
| Precision | 0.160 | **0.550** | — |
| Recall | 0.006 | **0.472** | — |
| Counting MAE | 203.7 | **49.1** | **75.9% reduction** |

Fine-tuning on only 31 shelf images produced large gains, with the biggest counting-error reductions on high-density shelves.

## Contents

```
RESULTS_AND_CONCLUSIONS.md   # Full write-up
data.yaml                     # YOLO dataset config
results/                      # Plots and CSVs
  density_vs_error.png
  mae_by_density_bar.png
  per_image_results.csv
  summary_by_density.csv
predictions/                  # Example inference visualizations
  pretrained/
  finetuned/
scripts/
  convert_supermarket_shelves.py
  evaluate_two_models.py
```

## Dataset

- Source: [Humans in the Loop Supermarket Shelves](https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/)
- 45 images, ~10k product instances
- Converted to YOLO format (Product class only)
- Split: 31 train / 7 val / 7 test (stratified by product count)

## Model

Lightweight YOLOv8n used as a CPU-friendly proxy for YOLOX-S.  
Fine-tuned for 10 epochs (batch=2, imgsz=640) on CPU.

## Key Takeaway

Domain shift from generic COCO pretraining to retail shelves is severe. Modest in-domain fine-tuning recovers usable detection and counting performance, especially on crowded shelves.
