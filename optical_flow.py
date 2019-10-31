import cv2
import numpy as np

class OpticalFlow:

    def __init__(self, frame):
        
        self.points = None
        self.st = None
        self.err = None

        frame_new = frame.copy()
        self.frame_old_gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        self.frame_new_gray = None
        self.p = None
        self.lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def track(self, frame):
        '''Input of boxes. Extract midpoints'''
        for box in self.boxes:
            b = np.array( [[box[0] + box[2], box[1] + box[3]]], dtype = np.float32 )
            self.p = b

        frame_new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lk_params = dict( winSize  = (15,15),   
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        frame_old = self.frame_old_gray.copy()

        p_new, status, error = cv2.calcOpticalFlowPyrLK(frame_old, frame_new_gray, b, None, **lk_params)
        self.point = p_new

        # Set this frame to the old one
        self.frame_old_gray = frame_new_gray.copy()

        return p_new

    def update_frame(self, frame):
        self.frame_old_gray = self.frame_new_gray
        frame_new = frame.copy()
        self.frame_new_gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)

    def track(self, frame):
        p_new, status, error = cv2.calcOpticalFlowPyrLK(self.frame_old_gray, self.frame_new_gray, self.points, None, **self.lk_params)
        self.points = p_new