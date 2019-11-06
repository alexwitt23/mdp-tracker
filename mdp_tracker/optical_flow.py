import cv2
import numpy as np


class OpticalFlow:

    def __init__(self, frame):
    
        # Gpu status 
        self.st = None
        # Error in the estimation
        self.err = None
        # Params for Lucas-Kanade Optical Flow
        self.lk_params = dict( winSize  = (9,9),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,8000, 0.03))

        # Original boxes to track
        self.boxes = []
        # Hold center points of input boxes
        self.center_points = None

        # Get copy of frame
        frame_new = frame.copy()
        self.frame_old_gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        self.frame_new_gray = None

    def update_cur_bbox_vec(self, boxes):
        """ Replace the list of boxes/points """
        self.boxes = boxes.copy()
        # Empty array initalization
        centers = np.empty( [len(self.boxes),2],dtype = np.float32 )
        # Populate array
        for idx, box in enumerate(self.boxes):
            # add in center points
            centers[idx] = np.array( [[box[0] + (box[2]/2), box[1] + (box[3]/2)]], dtype = np.float32 )

        self.center_points = centers

    def tracking_flow(self):
        """ Perform optical flow calc """
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.frame_old_gray, self.frame_new_gray, self.center_points, None, **self.lk_params)
        
        results = []
        # Loop over returned points to match with bbox
        for i, curr_point in enumerate(new_points):

            # old point
            prev_point = self.boxes[i]

            moved_x = curr_point[0] - prev_point[0]
            moved_y = curr_point[1] - prev_point[1]
            # Adjust positions for yolo format
            x_new = self.boxes[i][0] + moved_x - (self.boxes[i][2] / 2)
            y_new = self.boxes[i][1] + moved_y - (self.boxes[i][3] / 2)
            
            self.boxes[i] = (int(x_new), int(y_new),  self.boxes[i][2], self.boxes[i][3])

            results.append(self.boxes[i])
            self.points = results

        return results

    def update_frame(self, frame):

        self.frame_old_gray = self.frame_new_gray
        frame_new = frame.copy()
        self.frame_new_gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    def display_tracked_boxes(self, frame):

        for idx, box in enumerate(self.points):
            (x, y) = (box[0], box[1])
            (w, h) = (box[2], box[3])
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 3)
            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame
