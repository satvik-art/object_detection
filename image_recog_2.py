import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")  #make sure to download these files

# Load COCO class names with English labels
with open("coco.names", "r") as f:         #make sure to download coco.names
    classes = f.read().strip().split("\n")

def process_webcam():
    cap = cv2.VideoCapture(0)  # Access the default webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outs = net.forward(layer_names)

        # Process and draw bounding boxes
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x, center_y, width, height = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)

                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)
        key = cv2.waitKey(1)

        # Press 'q' to exit
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_still_image(image_path):
    frame = cv2.imread(image_path)

    # Preprocess the image for object detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    # Process and draw bounding boxes
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, width, height = (detection[:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)
        key = cv2.waitKey(0)

        # Press 'q' to exit
        if key & 0xFF == ord('q'):
            cv2.destroyAllWindows()

# Main program
mode = input("Choose mode (1 for webcam, 2 for still image): ")

if mode == "1":
    process_webcam()
elif mode == "2":
    image_path = input("Enter image file path: ")
    process_still_image(image_path)
else:
    print("Invalid mode. Choose 1 for webcam or 2 for still image.")
