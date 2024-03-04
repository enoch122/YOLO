import cv2
from ultralytics import YOLO

def process_video(video_path, model_path, output_width, output_height, confScore):
    # Initialize YOLO model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    outputStec = []

    # Process video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model.predict(frame, verbose=False, conf=confScore)

            # Display detected results with confidence scores and class id
            if len(results) > 0:
                for r in results:
                    if r.boxes.cls.numpy().size > 0 and r.boxes.conf.numpy().size > 0:
                        #can delete the following if working
                        print(f"Detected class id: {int(r.boxes.cls.numpy()[()][0])}; Confidence: {float(r.boxes.conf.numpy()[()][0])}")
                        
                        outputStec.append((int(r.boxes.cls.numpy()[()][0]), float(r.boxes.conf.numpy()[()][0])))

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    return outputStec

# process_video("C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\train\\IMG_0748.mp4", 'C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\train\\runs\\detect\\train\\weights\\best.pt',1200, 950, 0.7)
