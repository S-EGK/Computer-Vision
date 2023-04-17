# Computer Vision Projects

## Movenet Multipose Demo

This project is a Python implementation of a real-time human pose estimation system using a pre-trained deep learning model called Movenet Multipose. The system uses a camera feed to detect human poses in real-time video, visualizes the pose detections using OpenCV and Matplotlib libraries, and runs on GPUs for faster computations. The code requires proficiency in deep learning, computer vision, TensorFlow, Python programming, data visualization, and GPU computing.

## Face Detector

This project demonstrates how to use a pre-trained deep learning model for face detection in images and videos using Python and OpenCV. The 'Detector' class initializes the model and provides methods to process images and videos. The code highlights faces in the input by detecting them using the model and draws bounding boxes around them. This implementation of face detection can be useful in various applications such as video surveillance, facial recognition, and face tracking. The code requires the installation of OpenCV and NumPy libraries and uses the Caffe framework for the pre-trained model.

## Sign Language Recognition with CNN

This project builds and trains a Convolutional Neural Network (CNN) to classify sign language gestures from the Sign Language MNIST dataset. The dataset contains images of sign language gestures, represented as grayscale images of size 28x28 pixels. The code first loads and pre-processes the data, hot one encoding the labels and splitting the data into training and testing sets. The CNN model consists of three convolutional layers with pooling, followed by two fully connected layers with dropout. The model is compiled with the categorical cross-entropy loss function and the Adam optimizer. Finally, the model is trained on the training set and evaluated on the testing set, and the accuracy is plotted over the epochs. The trained model is saved for future use.
