from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print("YOLO11n baseline started")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")
    print("Model loaded")

    results = model.train(
        data="DSPCBSD+-1/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project="runs/detect/pcb_runs",
        name="yolo11n_baseline",
        workers=4,
        patience=10,
        amp=False,
        cache=False,
        plots=True
    )

    print("Training finished")