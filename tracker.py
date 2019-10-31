import argparse
import imutils
import time
import cv2
import sys
import numpy as np
 
# Opencv 3.x

def select_boxes(boxes):
	''' Wait for first boxes to be choosen '''
	# Read in boxes to track
	while True:
		ans = int(input("Enter the ID's of boxes to be tracked: "))
		if ans == -1:
			break
		else:
			boxes_to_track.append( boxes[ans] )

	return boxes_to_track

def initialize_tracker(boxes, OPENCV_OBJECT_TRACKERS, type):
	''' Initialize tracker on input boxes '''
	print("initializing")
	print(boxes)
	trackers = []
	for box in boxes:
		tracker = OPENCV_OBJECT_TRACKERS[str(type)]()
		print(box)
		(x, y) = (box[0], box[1])
		(w, h) = (box[2], box[3])
		box_init = (x, y, w, h)
		ok_init = tracker.init(frame_track, box_init)
		print("initialized:", ok_init)
		trackers.append(tracker)
	return trackers


if __name__ == '__main__' :


	net = cv2.dnn.readNetFromDarknet('./model/yolov3-tiny.cfg','./model/yolov3-tiny.weights')
	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	
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
	tracker = OPENCV_OBJECT_TRACKERS["boosting"]()

	# Read video
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
			for tracker in trackers:
				ok, box = tracker.update(frame)
				# Plot if found
				if ok:
					(x, y, w, h) = [int(v) for v in box]
					cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# Break if bad frame
		if not ok:
			break

		key = cv2.waitKey(1) & 0xFF
		
		# Make sure we only do detection every .01s
		if ((cv2.getTickCount() - timer) / cv2.getTickFrequency()) > TIME and not pause: 

			timer = cv2.getTickCount() 
			(H, W) = frame.shape[:2]
			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities

			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			boxes = []
			confidences = []
			classIDs = []

			# loop over each of the layer outputs. We only care about one detection that is human
			for output in layerOutputs:

				for detection in output:
						# extract the class ID and confidence (i.e., probability) of
						# the current object detection
						scores = detection[5:]
						classID = np.argmax(scores)
						confidence = scores[classID]

						# Look for humans
						if classID == 0:
							# filter out weak predictions by ensuring the detected
							# probability is greater than the minimum probability
							if confidence > .30:
								# scale the bounding box coordinates back relative to the
								# size of the image, keeping in mind that YOLO actually
								# returns the center (x, y)-coordinates of the bounding
								# box followed by the boxes' width and height
								box = detection[0:4] * np.array([W, H, W, H])
								(centerX, centerY, width, height) = box.astype("int")
					
								# use the center (x, y)-coordinates to derive the top and
								# and left corner of the bounding box
								x = int(centerX - (width / 2))
								y = int(centerY - (height / 2))
					
								# update our list of bounding box coordinates, confidences,
								# and class IDs
								boxes.append([x, y, int(width), int(height)])
								confidences.append(float(confidence))

			idxs = cv2.dnn.NMSBoxes(boxes, confidences, .30, .1)

		# Calculate Frames per second (FPS)
		# fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

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
				# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				# draw a bounding box rectangle and label on the image
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 3)
				cv2.putText(frame, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			cv2.imshow("Tracking", frame)
			cv2.waitKey(1)

		cv2.imshow("Tracking", frame)
		
		# Press c to choose boxes
		if pause and len(boxes) != 0:

			boxes_to_track = select_boxes(boxes)

			if len(boxes_to_track) != 0:
				pause = not pause
			
			# Initialize trackers
			trackers = initialize_tracker(boxes_to_track, OPENCV_OBJECT_TRACKERS, "boosting")