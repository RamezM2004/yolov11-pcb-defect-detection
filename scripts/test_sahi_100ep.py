from pathlib import Path
import yaml
import csv
import numpy as np
import pandas as pd
from PIL import Image

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


# ============================================================
# USER SETTINGS
# ============================================================

MODEL_PATH = Path(r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\MI 2\Project\100epchs_best\best.pt")
DATA_YAML = Path(r"DSPCBSD+-1\data.yaml")

TEST_IMAGES_DIR = Path(r"DSPCBSD+-1\test\images")
TEST_LABELS_DIR = Path(r"DSPCBSD+-1\test\labels")

OUTPUT_DIR = Path(r"sahi_results\best_100ep_sahi_full_metrics")

DEVICE = "cuda:0"

# Use low confidence for AP calculation, because AP needs predictions across confidence range
SAHI_MODEL_CONF = 0.001

# This confidence is used for Precision / Recall / F1 summary
METRIC_CONF = 0.25

# SAHI slicing settings
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# IoU thresholds like YOLO mAP50-95
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]

    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    return names


def xywhn_to_xyxy(box, img_w, img_h):
    """
    YOLO label format:
    class x_center_norm y_center_norm width_norm height_norm
    """
    x, y, w, h = box
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return [x1, y1, x2, y2]


def read_yolo_label(label_path, img_w, img_h):
    gt_boxes = []

    if not label_path.exists():
        return gt_boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = int(float(parts[0]))
        box = list(map(float, parts[1:5]))
        xyxy = xywhn_to_xyxy(box, img_w, img_h)

        gt_boxes.append({
            "class_id": cls,
            "box": xyxy
        })

    return gt_boxes


def box_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def compute_ap(recall, precision):
    """
    COCO-style 101-point interpolation AP.
    """
    recall = np.asarray(recall)
    precision = np.asarray(precision)

    if recall.size == 0:
        return 0.0

    recall_levels = np.linspace(0, 1, 101)
    ap = 0.0

    for r in recall_levels:
        p = precision[recall >= r]
        ap += p.max() if p.size > 0 else 0.0

    return ap / 101


def evaluate_ap_for_class(preds, gts, class_id, iou_threshold):
    """
    AP for one class at one IoU threshold.
    """
    class_preds = [p for p in preds if p["class_id"] == class_id]
    class_gts = [g for g in gts if g["class_id"] == class_id]

    n_gt = len(class_gts)

    if n_gt == 0:
        return None

    class_preds = sorted(class_preds, key=lambda x: x["confidence"], reverse=True)

    matched_gt = set()
    tp = []
    fp = []

    for pred in class_preds:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(class_gts):
            if gt["image_id"] != pred["image_id"]:
                continue

            if gt_idx in matched_gt:
                continue

            iou = box_iou(pred["box"], gt["box"])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp.append(1)
            fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0)
            fp.append(1)

    if len(tp) == 0:
        return 0.0

    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recall = tp_cumsum / (n_gt + 1e-9)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    ap = compute_ap(recall, precision)

    return ap


def evaluate_prf_at_conf(preds, gts, class_names, conf_threshold=0.25, iou_threshold=0.50):
    """
    Calculates Precision / Recall / F1 at fixed confidence and IoU=0.50.
    """
    rows = []

    filtered_preds = [p for p in preds if p["confidence"] >= conf_threshold]

    for class_id, class_name in enumerate(class_names):
        class_preds = [p for p in filtered_preds if p["class_id"] == class_id]
        class_gts = [g for g in gts if g["class_id"] == class_id]

        matched_gt = set()
        tp = 0
        fp = 0

        class_preds = sorted(class_preds, key=lambda x: x["confidence"], reverse=True)

        for pred in class_preds:
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(class_gts):
                if gt["image_id"] != pred["image_id"]:
                    continue

                if gt_idx in matched_gt:
                    continue

                iou = box_iou(pred["box"], gt["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn = len(class_gts) - tp

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visuals_dir = OUTPUT_DIR / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(DATA_YAML)
    num_classes = len(class_names)

    print("Class names:", class_names)
    print("Loading SAHI YOLO model...")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(MODEL_PATH),
        confidence_threshold=SAHI_MODEL_CONF,
        device=DEVICE,
    )

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(TEST_IMAGES_DIR.glob(ext))

    image_paths = sorted(image_paths)

    print(f"Found {len(image_paths)} test images.")

    all_predictions = []
    all_ground_truths = []

    for idx, image_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] Processing: {image_path.name}")

        image = Image.open(image_path)
        img_w, img_h = image.size

        label_path = TEST_LABELS_DIR / f"{image_path.stem}.txt"
        gt_boxes = read_yolo_label(label_path, img_w, img_h)

        for gt in gt_boxes:
            all_ground_truths.append({
                "image_id": image_path.name,
                "class_id": gt["class_id"],
                "box": gt["box"]
            })

        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
            verbose=0,
        )

        # Save visual prediction image
        try:
            result.export_visuals(export_dir=str(visuals_dir), file_name=image_path.stem)
        except Exception as e:
            print(f"Warning: could not save visual for {image_path.name}: {e}")

        for obj in result.object_prediction_list:
            bbox = obj.bbox

            pred_box = [
                float(bbox.minx),
                float(bbox.miny),
                float(bbox.maxx),
                float(bbox.maxy)
            ]

            pred_class = int(obj.category.id)
            pred_conf = float(obj.score.value)

            all_predictions.append({
                "image_id": image_path.name,
                "class_id": pred_class,
                "confidence": pred_conf,
                "box": pred_box
            })

    print("\nCalculating Precision / Recall / F1...")
    prf_rows = evaluate_prf_at_conf(
        all_predictions,
        all_ground_truths,
        class_names,
        conf_threshold=METRIC_CONF,
        iou_threshold=0.50
    )

    print("Calculating AP50 and mAP50-95...")

    ap_table = []

    for class_id, class_name in enumerate(class_names):
        class_ap_values = {}

        for iou_thr in IOU_THRESHOLDS:
            ap = evaluate_ap_for_class(
                all_predictions,
                all_ground_truths,
                class_id,
                iou_threshold=float(iou_thr)
            )

            class_ap_values[f"AP{int(iou_thr * 100)}"] = ap

        valid_aps = [v for v in class_ap_values.values() if v is not None]

        ap50 = class_ap_values["AP50"]

        if len(valid_aps) > 0:
            map_50_95 = float(np.mean(valid_aps))
        else:
            map_50_95 = None

        ap_table.append({
            "class_id": class_id,
            "class_name": class_name,
            "AP50": ap50,
            "mAP50-95": map_50_95,
            **class_ap_values
        })

    prf_df = pd.DataFrame(prf_rows)
    ap_df = pd.DataFrame(ap_table)

    final_df = prf_df.merge(ap_df, on=["class_id", "class_name"], how="left")

    metrics_csv = OUTPUT_DIR / "sahi_metrics_classwise.csv"
    final_df.to_csv(metrics_csv, index=False)

    preds_csv = OUTPUT_DIR / "sahi_predictions.csv"
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"])

        for p in all_predictions:
            cid = p["class_id"]
            cname = class_names[cid] if cid < len(class_names) else str(cid)
            writer.writerow([
                p["image_id"],
                cid,
                cname,
                p["confidence"],
                p["box"][0],
                p["box"][1],
                p["box"][2],
                p["box"][3],
            ])

    # Overall metrics
    total_tp = final_df["TP"].sum()
    total_fp = final_df["FP"].sum()
    total_fn = final_df["FN"].sum()

    overall_precision = total_tp / (total_tp + total_fp + 1e-9)
    overall_recall = total_tp / (total_tp + total_fn + 1e-9)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-9)

    valid_ap50 = final_df["AP50"].dropna()
    valid_map = final_df["mAP50-95"].dropna()

    overall_map50 = valid_ap50.mean() if len(valid_ap50) > 0 else 0.0
    overall_map50_95 = valid_map.mean() if len(valid_map) > 0 else 0.0

    summary_path = OUTPUT_DIR / "summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAHI YOLO11s Full Metrics Summary\n")
        f.write("=================================\n\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Images: {TEST_IMAGES_DIR}\n")
        f.write(f"Labels: {TEST_LABELS_DIR}\n")
        f.write(f"Slice size: {SLICE_WIDTH} x {SLICE_HEIGHT}\n")
        f.write(f"Overlap: {OVERLAP_WIDTH_RATIO}, {OVERLAP_HEIGHT_RATIO}\n")
        f.write(f"Metric confidence threshold: {METRIC_CONF}\n\n")

        f.write(f"Overall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall:    {overall_recall:.4f}\n")
        f.write(f"Overall F1:        {overall_f1:.4f}\n")
        f.write(f"Overall mAP50:     {overall_map50:.4f}\n")
        f.write(f"Overall mAP50-95:  {overall_map50_95:.4f}\n\n")

        f.write("Class-wise metrics saved to sahi_metrics_classwise.csv\n")
        f.write("Predictions saved to sahi_predictions.csv\n")

    print("\n==============================")
    print("SAHI Evaluation Finished")
    print("==============================")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall:    {overall_recall:.4f}")
    print(f"Overall F1:        {overall_f1:.4f}")
    print(f"Overall mAP50:     {overall_map50:.4f}")
    print(f"Overall mAP50-95:  {overall_map50_95:.4f}")
    print()
    print(f"Saved class-wise metrics to: {metrics_csv}")
    print(f"Saved predictions to:        {preds_csv}")
    print(f"Saved summary to:            {summary_path}")
    print(f"Saved visuals to:            {visuals_dir}")


if __name__ == "__main__":
    main()