# Two-Model Experiment: Pretrained vs. Fine-tuned on Supermarket Shelves

**Dataset:** Humans in the Loop Supermarket Shelves (45 images, ~10k product instances)  
**Proxy model:** Ultralytics YOLOv8n (lightweight stand-in for YOLOX-S; CPU-only training)  
**Split:** 31 train / 7 val / 7 test (stratified by product count)

## Detection Metrics (Test Set, official Ultralytics val)

| Model          | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|----------------|-----------|--------|---------|--------------|
| Pretrained (COCO) | 0.160    | 0.006  | 0.005   | 0.002        |
| Fine-tuned     | **0.550** | **0.472** | **0.442** | **0.219**   |

Fine-tuning yields a **~88× increase in mAP@0.5** relative to the COCO-pretrained baseline.

## Counting Metrics (Test Set)

| Model          | Overall MAE | Mean signed error |
|----------------|-------------|-------------------|
| Pretrained     | 203.7       | large under-count |
| Fine-tuned     | **49.1**    | much closer       |

**MAE reduction: 75.9%**

### Stratified by Density Tier

Density thresholds (test-set tertiles): low ≤ 144 products, medium ≤ 259, high > 259.

| Density | Mean GT count | Pretrained MAE | Fine-tuned MAE | MAE reduction |
|---------|---------------|----------------|----------------|---------------|
| Low     | 47.5          | high           | lower          | 53.8%         |
| Medium  | 167.0         | high           | lower          | 41.8%         |
| High    | 342.7         | very high      | substantially lower | **88.5%** |

The largest absolute and relative gains occur on the highest-density shelves, where the pretrained model essentially fails to detect products.

## Visualizations

- `results/density_vs_error.png` — absolute counting error vs. ground-truth product count
- `results/mae_by_density_bar.png` — MAE by density tier for both models

## Conclusions

1. **Domain shift is severe.** A COCO-pretrained detector (even a modern YOLOv8n) produces near-zero recall on retail shelf products. Generic object detectors do not transfer to densely packed, small-scale SKUs without adaptation.

2. **Targeted fine-tuning is highly effective.** Ten epochs of fine-tuning on only 31 shelf images raised mAP@0.5 from 0.005 to 0.442 and cut counting MAE by 76%. The model learns the visual characteristics of products on shelves.

3. **Density matters.** Absolute counting error grows with product density for both models, but fine-tuning disproportionately helps the high-density regime (88% MAE reduction). This supports the original proposal’s emphasis on density-stratified evaluation: crowded shelves are both harder and the place where domain adaptation delivers the most value.

4. **Practical takeaway for retail shelf monitoring.** Even a lightweight detector, when fine-tuned on a modest amount of in-domain data, becomes usable for product counting and density estimation. Further gains are expected from longer training, higher resolution, or a true YOLOX-S architecture on a larger set such as SKU-110K.

**Artifacts**
- Converted YOLO dataset: `dataset/`
- Fine-tuned weights: `runs/detect/runs/finetuned_yolov8n/weights/best.pt`
- Per-image results: `results/per_image_results.csv`
- Summary: `results/summary_by_density.csv`
