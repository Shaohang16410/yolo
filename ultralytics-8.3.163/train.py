from ultralytics import YOLO

model = YOLO(r"yolov8n.pt")

model.predict(
    source=r"ultralytics/assets",
    save=True,
    show=False,
)