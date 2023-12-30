from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO('yolov8n.pt')

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(data='C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\data.yaml', epochs=50, iterations=20, optimizer='Adam', plots=False, save=False, val=False)