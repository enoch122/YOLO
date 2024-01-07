from ultralytics import YOLO
def main():
    print("================================")
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data='C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\dataset\\data.yaml', epochs=300, patience=30, device=0, optimizer="Adam", val=False)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # path = model.export()  # export the model to .pt format
if __name__ == '__main__':
    main()