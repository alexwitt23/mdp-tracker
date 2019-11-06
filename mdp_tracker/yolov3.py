import cv2
import numpy as np

class yolov3:

    def __init__(self, cfg, weights, H, W):

        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        layerNames = self.net.getLayerNames()
        self.layers = [layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.H = H
        self.W = W
        self.boxes = []
        
    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # Forward pass through model
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.layers)

        boxes = []
        confidences = []
        classIDs = []

        nms_boxes = []

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
                            box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                            (centerX, centerY, width, height) = box.astype("int")
                
                            # use the center (x, y)-coordinates to derive the top and
                            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))
                
                            # update our list of bounding box coordinates, confidences,
                            # and class IDs
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            
                            indices = cv2.dnn.NMSBoxes(boxes, confidences, .2, .2)

                            for i in indices:
                                idx = i[0]
                                nms_boxes.append((boxes[idx][0],boxes[idx][1],boxes[idx][2],boxes[idx][3]))
                            
                            self.boxes = nms_boxes

        return nms_boxes

    def draw_boxes(self, frame):

        for idx, box in enumerate(self.boxes):
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 3)
            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return frame
