#!/usr/bin/env python
import numpy as np
import cv2

class dense_of:

    def __init__(self, frame, height_of, width_of, h_ori, w_ori):
        self.height_ori = h_ori
        self.width_ori = w_ori

        frame_new = cv2.resize(frame, (width_of, height_of), interpolation = cv2.INTER_AREA)
        self.prevgray = gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        self.height_of = height_of
        self.width_of = width_of

        self.boxes = []

    def initialize_boxes(self, boxes):
        self.boxes = boxes

    def calc_flow(self, frame):
        img = cv2.resize(frame,(self.width_of, self.height_of), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prevgray = gray
        
        return flow 

    def update_tracker_flow(self, frame):
        flow = self.calc_flow(frame)
        vx = flow[...,0] 
        vy = flow[...,1]

        boxes = []
        boxes_n = []
        # Loop over boxes
        for box in self.boxes:

            (x, y) = (int(box[0]*self.width_of), int(box[1]*self.height_of))
            (w, h) = (int(box[2]*self.width_of), int(box[3]*self.height_of))

            box_of = (x, y, w, h)

            # Loop over the x,y velocity field in this bbox
            vx_sum = 0.0
            vy_sum = 0.0
            counter = 0 
            for i in range(x, (x + w - 1)):
                for j in range(y, (y + h - 1)):
                    if j < self.height_of and i < self.width_of:
                        vx_sum += vx[j,i]
                        vy_sum += vy[j,i]
                        counter += 1

            # Normalized
            vx_n = (vx_sum / counter) / self.width_of
            vy_n = (vy_sum / counter) / self.height_of 
            # Optical flow coords
            vx_of = vx_n * self.width_of
            vy_of = vx_n * self.height_of

            boxes.append((int((x + vx_of)*self.width_ori/self.width_of), 
                          int((y + vy_of)*self.height_ori/self.height_of), 
                          int(w*self.width_ori/self.width_of), 
                          int(h*self.height_ori/self.height_of)))

            boxes_n.append(((x/self.width_of) + vx_n, (y/self.height_of) + vy_n, w / self.width_of, h / self.height_of))
        
        self.boxes = boxes_n
        
        return boxes
