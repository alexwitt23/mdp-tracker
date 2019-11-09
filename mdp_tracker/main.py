import argparse
import imutils
import time
import cv2
import sys
import numpy as np

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

def initialize_tracker(boxes, OPENCV_OBJECT_TRACKERS, type):
	''' Initialize tracker on input boxes '''
	trackers = []
	for box in boxes:

		tracker = OPENCV_OBJECT_TRACKERS[str(type)]()
		(x, y) = (box[0], box[1])
		(w, h) = (box[2], box[3])
		box_init = (x, y, w, h)
		ok_init = tracker.init(frame_track, box_init)
		print("initialized:", ok_init)
		trackers.append(tracker)

	return trackers

def calculateIntersection(rect1, rect2):
	# rect = (x,y,w,h)
	x1, y1, w1, h1 = rect1
	x2, y2, w2, h2 = rect2

    if x1 >= x2 and (x1+w1) <= (x2+w2) : # Contained
        intersection = (x1+w1) - x1

    elif x1 < x2 and (x1+w1) > (x2+w2): # Contains
        intersection = b1 - b0

    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0

    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
		
    else: # No intersection (either side)
        intersection = 0

    return intersection


if __name__ == '__main__' :

	# Dictionary of OpenCV trackers
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}
 
	# Choose and create tracker
	tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
	# Colors for tracking boxes
	colors = np.random.randint(0,255,(100,3))

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

	# Start timer
	timer = cv2.getTickCount()
	# Box to track
	trackers = []
	# Output of yolo detection
	boxes = []

	TIME = .04
	ok_init = False
	idxs = []

	choose_boxes = False
	stop = True
	boxes_to_track = []
	pause = False
	# Loop over video
	while True:

		# Read a new frame
		if not pause:
			ok, frame = video.read()

		# Do tracking 
		if len(trackers) != 0:
			rects = []
			for idx, tracker in enumerate(trackers):
				ok, box = tracker.update(frame)
				# Plot if found
				if ok:
					(x, y, w, h) = [int(v) for v in box]
					cv2.rectangle(frame, (x, y), (x + w, y + h), colors[idx].tolist(), 2)
					rects.append((x, y, w, h))

			for idx1, rect1 in enumerate(rects):
				for idx2, rect2 in enumerate(rects):
					# if not same rect, check intersection
					if (idx1 != idx2):
						calculateIntersection(rect1, rect2)

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
		if pause and len(boxes) != 0:
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
		if pause and len(boxes) != 0:

			boxes_to_track = select_boxes(boxes)

			if len(boxes_to_track) != 0:
				# Initialize trackers
				trackers = initialize_tracker(boxes_to_track, OPENCV_OBJECT_TRACKERS, "boosting")
			
			pause = not pause