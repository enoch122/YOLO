import cv2
from ultralytics import YOLO

model = YOLO('C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\train\\runs\\detect\\train\\weights\\best.pt')

video_path = "C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\test img\\test video\\IMG_0748.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (1500, 1250))  
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()