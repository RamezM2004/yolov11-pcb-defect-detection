from ultralytics import YOLO
import torch
from pathlib import Path

# ============================================================
# Experiment B: YOLO11s + CBAM + Oversampled Dataset
# 25 epochs, batch=8, no nominal batch accumulation
# ============================================================

PROJECT_ROOT = Path(__file__).parent

CBAM_YAML = PROJECT_ROOT / "custom_models" / "yolo11s_cbam.yaml"
DATA_YAML = PROJECT_ROOT / "DSPCBSD+-1" / "data.yaml"

BEST_WEIGHTS = PROJECT_ROOT / "100epchs_best" / "best.pt"

if __name__ == "__main__":

    print("=" * 70)
    print("Experiment B: YOLO11s + CBAM + Oversampled Training Dataset")
    print("25 epochs, batch=8, no accumulation")
    print("=" * 70)

    print(f"CBAM YAML:     {CBAM_YAML}")
    print(f"Dataset YAML:  {DATA_YAML}")
    print(f"Start weights: {BEST_WEIGHTS}")

    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    if not CBAM_YAML.exists():
        raise FileNotFoundError(f"CBAM YAML not found: {CBAM_YAML}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    if not BEST_WEIGHTS.exists():
        raise FileNotFoundError(f"Best weights not found: {BEST_WEIGHTS}")

    # Build YOLO11s + CBAM architecture
    model = YOLO(str(CBAM_YAML))

    # Load matching YOLO11s baseline layers.
    # CBAM layers remain randomly initialized.
    model.load(str(BEST_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=640,
        batch=8,
        device=0,

        project=str(PROJECT_ROOT / "yolo11s_custom_results"),
        name="expB_yolo11s_cbam_oversampled_25ep_b8",

        workers=2,
        cache=False,
        plots=True,

        # GTX 1650: keep AMP disabled
        amp=False,

        # Train full network
        freeze=0,

        # Optimizer
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.01,
        cos_lr=True,
        weight_decay=0.0005,
        patience=10,

        # No accumulation / no nominal batch scaling
        # Real batch = 8, nominal batch = 8
        nbs=8,

        # Normal YOLO loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Mild augmentation for PCB defects
        degrees=2.0,
        translate=0.02,
        scale=0.10,
        shear=0.0,
        perspective=0.0,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=5,

        hsv_h=0.01,
        hsv_s=0.20,
        hsv_v=0.10,

        erasing=0.0,

        seed=0,
        deterministic=True,
        save=True,
        exist_ok=False,
    )

    print("\nExperiment B finished.")