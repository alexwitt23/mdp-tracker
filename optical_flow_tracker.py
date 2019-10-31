import argparse
import imutils
import time
import cv2
import sys
import numpy as np

from optical_flow import OpticalFlow
from yolov3 import yolov3


def select_boxes(boxes):
	''' Wait for first boxes to be choosen '''
	# Read in boxes to track
	boxes_to_track = []
	while True:
		ans = int(input("Enter the ID's of boxes to be tracked: "))
		if ans == -1:
			break
		else:
			boxes_to_track.append( boxes[ans] )

	return boxes_to_track



if __name__ == '__main__' :

	# Read video
	#video = cv2.VideoCapture(0)
	video = cv2.VideoCapture("sample_video.mp4")

	# Exit if video not opened.
	if not video.isOpened():
		print("Could not open video")
		sys.exit()

	# Read first frame.
	ok, frame = video.read()
	if not ok:
		print('Cannot read video file')
		sys.exit()

	# Create Neural Net
	H, W = frame.shape[:2]
	nn = yolov3('./model/yolov3-tiny.cfg','./model/yolov3-tiny.weights', H, W)
	
	# Initialize optical flow tracker 
	optical_flow = OpticalFlow(frame)

	# Output of yolo detection
	boxes = []
	# Output of tracking
	boxes_track = []

	# Start timer
	timer = cv2.getTickCount()
	TIME = .5

	pause = False
	# Loop over video
	
	while True:

		c = cv2.waitKey(10)

		# Read a new frame
		if not pause:
			ok, frame = video.read()

		# Break if bad frame
		if not ok:
			break
		# Update Optical Flow frame
		optical_flow.update_frame(frame)

		# Make sure we only do detection every TIME
		if ((cv2.getTickCount() - timer) / cv2.getTickFrequency()) > TIME and not pause: 
			# Do YOLO Detection
			boxes = nn.detect(frame)
			boxes_track = optical_flow.track()
		
		# use yolo class to draw boxes
		frame = nn.draw_boxes(frame)

		cv2.imshow("Tracking", frame)
		if c == 113: #q
			break
		elif c == 112: #p
			# pause and select boxes to track
			boxes_to_track = select_boxes(boxes)
			optical_flow.update_tracker(boxes, frame)


