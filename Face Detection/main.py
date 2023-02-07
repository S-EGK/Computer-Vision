from Detector import *

detector = Detector(use_cuda=True)

detector.processImg("friends.jpg")
detector.processVideo("video.mp4")