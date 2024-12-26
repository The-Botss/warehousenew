from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolo11n.pt')  # YOLOv8s pre-trained model

# Train the model with your data.yaml
results = model.train(data='/home/ubuntu/warehouse-box/dataextended/data.yaml', epochs=100, imgsz=640)
