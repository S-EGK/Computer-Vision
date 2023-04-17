# Movenet Multipose Demo for Pose Estimation

This code provides a demo for real-time pose estimation using Movenet Multipose, which is a lightweight, single-shot detector model for human pose estimation.

The demo reads frames from the default camera and estimates the poses of the detected persons. The pose information includes the keypoint locations and the confidence scores, which can be used to draw the skeleton and keypoints on the image.

## Requirements
- Python 3
- TensorFlow 2.x
- TensorFlow Hub
- OpenCV
- Matplotlib
- NumPy

### Installation
- Install the required libraries using pip

```sh
pip install tensorflow tensorflow_hub opencv-python matplotlib numpy
```

## Usage
- Run the demo using the following command:

```sh
python movenet_multipose_demo.py
```
- The demo will open a window that shows the camera feed with the detected poses.

## Code Explanation
- The code first imports the necessary libraries and sets up the GPU memory growth for TensorFlow if available.
- It loads the Movenet Multipose model from TensorFlow Hub and extracts the serving signature.
- Two functions are defined to draw the keypoints and edges of the pose skeleton on the input image using OpenCV.
- The 'loop_through_people' function loops through each person detected in the input frame and renders the skeleton and keypoints for each person using the draw_connections and draw_keypoints functions.
- The 'cap' variable reads frames from the default camera using OpenCV and performs the following steps for each frame:
  - Resizes the frame and pads it to a fixed size for input to the model.
  - Passes the resized frame through the Movenet model to get the pose estimation results.
  - Calls the 'loop_through_people' function to render the poses on the frame.
  - Displays the resulting frame in a window using OpenCV.
  - Waits for a key press and breaks the loop if the key 'q' is pressed.

## Tech Stack

**Python** programming language

**TensorFlow** machine learning library and its related libraries like **TensorFlow Hub**

**OpenCV** computer vision library for image and video processing

**Matplotlib** library for data visualization

**NumPy** library for scientific computing and array processing.

## 🛠 Skills
- Deep Learning
- Computer Vision
- TensorFlow
- Data Visualization
- GPU Computing

## Authors
- [Srikanth Elkoori Ghantala Karnam](https://www.github.com/S-EGK)
