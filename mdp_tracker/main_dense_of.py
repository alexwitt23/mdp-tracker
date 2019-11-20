import argparse
import imutils
import time
import cv2
import sys
import numpy as np
from collections import namedtuple

from yolov3 import yolov3
from dense_optical_flow import dense_of

# Define rectangle tuple for easier iou calc
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def iou(a, b): 
	"""Get IOU of two boxes"""
	a1 = (a.xmin - a.xmax) * (a.ymin - a.ymax)
	a2 = (b.xmin - b.xmax) * (b.ymin - b.ymax)

	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
	
	if (dx>=0) and (dy>=0):
		return (dx*dy / (a1 + a2 - (dx*dy)))	

def select_boxes(boxes, H, W):
	"""Select detection boxes for tracking"""

	boxes_to_track = []

	while True:

		ans = int(input("Enter the ID's of boxes to be tracked: "))
		
		if ans == -1:
			break
		else:
			box = boxes[ans]
			(x, y) = (box[0] / W, box[1] / H)
			(w, h) = (box[2] / W, box[3] / H)

			box_n = (x, y, w, h)

			boxes_to_track.append( box_n )

	return boxes_to_track


if __name__ == '__main__' :

	# Colors for tracking boxes
	colors = np.random.randint(0,255,(50,3))

	# Read video or webcam
	if len(sys.argv) == 2:
		video = cv2.VideoCapture(sys.argv[1])
	else:
		video = cv2.VideoCapture(0)

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

	dof = dense_of(frame, 400, 600, H, W) 
	# Start timer
	timer = cv2.getTickCount()
	# Box to track
	trackers = []
	# Output of yolo detection
	boxes = []

	TIME = .04
	IOU_THRESH = .65

	boxes_to_track = []
	pause = False

	# Loop over video
	while True:

		# Read a new frame
		if not pause:
			ok, frame = video.read()

		# Update optical flow
		if len(dof.boxes) != 0:
			rects = []
			boxes = dof.update_tracker_flow(frame)
			for idx, box in enumerate(boxes):
				(x, y, w, h) = box
				color = (255, 0, 0) 
				cv2.rectangle(frame, (x, y), (x + w, y + h), colors[idx].tolist(), 2)
				rects.append(Rectangle(x, y, x+w, y+h))
			"""
			for idx1, rect1 in enumerate(rects):
				for idx2, rect2 in enumerate(rects):
					# if not same rect, check intersection
					_iou = iou(rect1, rect2)
					if idx1 != idx2 and (_iou != None) and _iou >= IOU_THRESH:
						trackers = []
						boxes_to_track = []
			"""

			
						

		# Break if bad frame
		if not ok:
			break

		key = cv2.waitKey(1) & 0xFF
		
		# Make sure we only do detection every .01s
		if ((cv2.getTickCount() - timer) / cv2.getTickFrequency()) > TIME and not pause: 
			# Do YOLO Detection
			boxes = nn.detect(frame)
			
		# Use waitkey
		if key == 32:
			pause = not pause

		# If hit SPACE, pause video for new tracker selection
		frame_track = None
		if pause != 0 and len(dof.boxes) == 0:
			# get clean copy of frame
			frame_track = frame 
			# pause while getting boxes to track
			'''Plot the detections'''
			# use yolo class to draw boxes
			frame = nn.draw_boxes(frame)
			cv2.imshow("Tracking", frame)
			cv2.waitKey(1)

		cv2.imshow("Tracking", frame)
		
		# Press c to choose boxes
		if pause != 0  and len(dof.boxes) == 0:
			boxes_to_track = select_boxes(boxes, H, W)
			dof.initialize_boxes(boxes_to_track)
			pause = not pause