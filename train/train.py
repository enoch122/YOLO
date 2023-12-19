from ultralytics import YOLO
def main():
    print("================================")
    model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data='C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\data.yaml', epochs=300, patience=50, device=0, optimizer="AdamW")  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    print("****************************************************************")
    train_result = model("C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\train\\images\\img_23102023181455039223_jpg.rf.e78e7f1d2e672bde7f01ac5d8c6ec4f1.jpg")  # predict on an image
    print("================================")
    # path = model.export()  # export the model to .pt format
if __name__ == '__main__':
    main()