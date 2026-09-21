from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================================
# Paths
# ============================================================

P2_CA_YAML = r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\MI 2\Project\custom_models\yolo11s_p2_ca.yaml"

DATA_YAML = r"DSPCBSD+-1\data.yaml"

P2_BEST = Path(
    r"C:\Users\USER\OneDrive - GJU\Desktop\4th Year\MI 2\Project\runs\detect\yolo11s_custom_results\yolo11s_p2_60ep_phase2_unfrozen\weights\best.pt"
)

if not P2_BEST.exists():
    raise FileNotFoundError(f"Could not find P2 best.pt: {P2_BEST}")

PROJECT_NAME = "yolo11s_custom_results"

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("YOLO11s + P2 + CA once + adjusted loss")
    print("Starting from P2 best weights:", P2_BEST)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ============================================================
    # Phase 1: Warm up CA module
    # ============================================================

    print("\n========== Phase 1: P2 + CA once warmup, 10 epochs ==========")

    model = YOLO(P2_CA_YAML)
    model.load(str(P2_BEST))

    model.train(
        data=DATA_YAML,
        epochs=10,
        imgsz=640,
        batch=4,
        device=0,

        project=PROJECT_NAME,
        name="yolo11s_p2_ca_once_loss_phase1",

        workers=2,
        cache=False,
        plots=True,

        amp=False,

        # Freeze original backbone/early features
        # CA is in neck/head, so it can still learn
        freeze=10,

        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        patience=8,

        nbs=16,

        # Loss adjustment
        box=8.5,
        cls=0.6,
        dfl=2.0,

        # No augmentation
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        fliplr=0.0,
        flipud=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        erasing=0.0,

        seed=0,
        deterministic=True,
        save=True,
        exist_ok=False,
    )

    phase1_best = Path("runs") / "detect" / PROJECT_NAME / "yolo11s_p2_ca_once_loss_phase1" / "weights" / "best.pt"

    if not phase1_best.exists():
        phase1_best = Path(PROJECT_NAME) / "yolo11s_p2_ca_once_loss_phase1" / "weights" / "best.pt"

    if not phase1_best.exists():
        raise FileNotFoundError(f"Could not find Phase 1 best.pt: {phase1_best}")

    print("Phase 1 best:", phase1_best)

    # ============================================================
    # Phase 2: Full fine-tuning
    # ============================================================

    print("\n========== Phase 2: P2 + CA once full fine-tuning, 40 epochs ==========")

    model = YOLO(str(phase1_best))

    model.train(
        data=DATA_YAML,
        epochs=40,
        imgsz=640,
        batch=4,
        device=0,

        project=PROJECT_NAME,
        name="yolo11s_p2_ca_once_loss_phase2",

        workers=2,
        cache=False,
        plots=True,

        amp=False,
        freeze=0,

        optimizer="AdamW",
        lr0=0.00005,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        patience=12,

        nbs=16,

        # Loss adjustment
        box=8.5,
        cls=0.6,
        dfl=2.0,

        # No augmentation
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,

        fliplr=0.0,
        flipud=0.0,

        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,

        erasing=0.0,

        seed=0,
        deterministic=True,
        save=True,
        exist_ok=False,
    )

    print("\nYOLO11s + P2 + CA once + loss-adjusted training finished.")