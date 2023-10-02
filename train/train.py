from ultralytics import YOLO
print("================================")
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# train_result = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
print("================================")
path = model.export()  # export the model to .pt format