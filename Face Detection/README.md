# Face Detector

This repository contains a Python script that uses OpenCV's Deep Neural Network (DNN) module to detect faces in an image or a video.

## Requirements

To run this script, you need:

- Python 3.x
- OpenCV 4.x
- NumPy

You can install OpenCV and NumPy using pip:

```sh
pip install opencv-python numpy
```

## Usage

To use the face detector, you need to create an instance of the Detector class and call one of its methods:

```sh
detector = Detector(use_cuda=True)

# Process an image
detector.processImg("friends.jpg")

# Process a video
detector.processVideo("video.mp4")
```

You can pass the use_cuda parameter to the Detector constructor to enable CUDA acceleration (if available).

## Class: Detector

### Method: __init__(self, use_cuda = False)

Creates a new instance of the Detector class.
- 'use_cuda' (optional, default=False): a boolean that indicates whether to use CUDA acceleration.

### Method: processImg(self, imgName)

Detects faces in a given image and displays the output.
- 'imgName': a string that contains the path to the image file.

### Method: processVideo(self, videoName)

Detects faces in a given video and displays the output.
- 'videoName': a string that contains the path to the video file.

### Method: processFrame(self)

Processes a single frame of the input video and detects faces. This method is called by the 'processVideo' method.

## Tech Stack

**Python** programming language

**OpenCV** library for computer vision tasks

**NumPy** library for numerical computing

**imutils** library for basic image processing functions


## 🛠 Skills
- Deep Learning
- Computer Vision
- Object Detection
- Image Processing
- Video Processing
- Convolutional neural networks (CNN)
- Face detection
- Caffe framework


## Authors
- [Srikanth Elkoori Ghantala Karnam](https://www.github.com/S-EGK)
