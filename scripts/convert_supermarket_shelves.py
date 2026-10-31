#!/usr/bin/env python3
"""Convert Supermarket Shelves annotations to YOLO format and split dataset."""
import json
import os
import shutil
import glob
from pathlib import Path
import numpy as np

SRC = Path("/workspace/attachments")
DST = Path("/workspace/artifacts/yolox_retail_shelf/real_shelf_experiment/dataset")

def parse_box(points, img_w, img_h):
    """Convert exterior points to normalized YOLO xywh."""
    if not points or "exterior" not in points:
        return None
    ext = points["exterior"]
    if len(ext) < 2:
        return None
    x1, y1 = ext[0]
    x2, y2 = ext[1]
    # Ensure x1 < x2, y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    # Clip
    x1 = max(0, min(img_w, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return xc, yc, w, h

def main():
    img_files = sorted(SRC.glob("*.jpg"))
    records = []
    
    for img_path in img_files:
        json_path = img_path.with_suffix(".jpg.json")
        if not json_path.exists():
            continue
        data = json.load(open(json_path))
        h = data["size"]["height"]
        w = data["size"]["width"]
        
        boxes = []
        for obj in data["objects"]:
            if obj.get("classTitle") != "Product":
                continue
            box = parse_box(obj.get("points"), w, h)
            if box:
                boxes.append(box)
        
        records.append({
            "img": img_path,
            "name": img_path.stem,
            "boxes": boxes,
            "count": len(boxes),
            "w": w,
            "h": h
        })
    
    # Sort by product count for stratified split
    records.sort(key=lambda r: r["count"])
    n = len(records)
    print(f"Total images with products: {n}")
    print(f"Product counts: min={records[0]['count']}, max={records[-1]['count']}, median={records[n//2]['count']}")
    
    # Split: ~70% train, 15% val, 15% test, stratified by interleaving
    train, val, test = [], [], []
    for i, r in enumerate(records):
        if i % 7 == 0:
            test.append(r)
        elif i % 7 == 1:
            val.append(r)
        else:
            train.append(r)
    
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    for split_name, split_recs in [("train", train), ("val", val), ("test", test)]:
        img_dir = DST / "images" / split_name
        lbl_dir = DST / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for r in split_recs:
            # Copy image
            shutil.copy(r["img"], img_dir / r["img"].name)
            # Write label
            lbl_path = lbl_dir / f"{r['name']}.txt"
            with open(lbl_path, "w") as f:
                for xc, yc, bw, bh in r["boxes"]:
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
    
    # data.yaml
    yaml_path = DST / "data.yaml"
    yaml_path.write_text(f"""path: {DST}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['product']
""")
    print(f"Wrote {yaml_path}")
    
    # Save density info
    import pandas as pd
    density_df = pd.DataFrame([{
        "image": r["name"],
        "split": next(s for s, lst in [("train", train), ("val", val), ("test", test)] if r in lst),
        "product_count": r["count"],
        "width": r["w"],
        "height": r["h"]
    } for r in records])
    density_df.to_csv(DST.parent / "results" / "image_density.csv", index=False)
    print("Saved density CSV")

if __name__ == "__main__":
    main()
