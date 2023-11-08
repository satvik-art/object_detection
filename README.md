# Object Detection with YOLO

This Python script uses the YOLO (You Only Look Once) deep learning model to perform real-time object detection on either a webcam feed or a still image. YOLO is known for its speed and accuracy in object detection tasks. It can detect a wide range of objects in images and videos.(Used an older version of YOLO, preferably use V6 or V7 if possible)

## Requirements
- Python 3.x
- OpenCV (cv2)
- Pre-trained YOLO weights and configuration files (downloadable from the official YOLO website)
- COCO class names file (provided in the repository)

## Installation
1. Install Python 3.x: [Python Download](https://www.python.org/downloads/)
2. Install OpenCV: `pip install opencv-python`

## Usage
You can choose between two modes: webcam and still image.

### Mode 1: Webcam
Run the script and select mode 1. It will access your default webcam and display real-time object detection results.

```python
python object_detection.py
Choose mode (1 for webcam, 2 for still image): 1
```

While running, press 'q' to exit the webcam mode.

### Mode 2: Still Image
Select mode 2, and provide the file path to an image you want to analyze. The script will display the image with bounding boxes around detected objects.

```python
python object_detection.py
Choose mode (1 for webcam, 2 for still image): 2
Enter image file path: your_image.jpg
```

While viewing the image, press 'q' key to close the window.

## Important Notes
- The YOLO weights and configuration files must be downloaded from the official YOLO website and specified in the script.
- The COCO class names file is included in the repository.

## Customization
You can modify the confidence threshold and non-maximum suppression threshold in the script to adjust the object detection sensitivity. Utilize this samll code snippet for further use cases.

## Author
Satvik Jalali

Feel free to customize and enhance this script for your specific needs. Happy object detection!
