from ultralytics import YOLO
print("================================")
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data='C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\data.yaml', epochs=100, patience=50)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
train_result = model("C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\train\\images\\IMG_0636_jpg.rf.5e35a83bc69a4a081ed09b48456f6384.jpg")  # predict on an image
print("================================")
# path = model.export()  # export the model to .pt format