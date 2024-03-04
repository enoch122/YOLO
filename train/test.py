import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\train\\runs\\detect\\train\\weights\\best.pt')

# Set video path
video_path = "C:\\Users\\enoch\\OneDrive\\桌面\\YOLO\\train\\IMG_0748.mp4"
cap = cv2.VideoCapture(video_path)

# Set output parameters
output_path = "output.mp4"
output_width = 1200
output_height = 950
output_fps = 20.0
output_codec = cv2.VideoWriter_fourcc(*"mp4v")

# Initialize video writer
output_writer = cv2.VideoWriter(output_path, output_codec, output_fps, (output_width, output_height))
outputStec = []


# Process video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.predict(frame, verbose=False, conf=0.7)
        
        # print(results[0].names)

        # Display detected results with confidence scores and class id
        if len(results) > 0:
            for r in results:
                if r.boxes.cls.numpy().size > 0 and r.boxes.conf.numpy().size > 0:
                    print(f"Detected class id: {int(r.boxes.cls.numpy()[()][0])}; Confidence: {float(r.boxes.conf.numpy()[()][0])}")
                    outputStec.append((int(r.boxes.cls.numpy()[()][0]), float(r.boxes.conf.numpy()[()][0])))
        # Visualize the results on the frame
            annotated_frame = results[0].plot()
            annotated_frame = cv2.resize(annotated_frame, (output_width, output_height))

        # Write the annotated frame to the output video file
            output_writer.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, output writer, and close the display window
cap.release()
output_writer.release()
cv2.destroyAllWindows()
print(outputStec)
print(len(outputStec))