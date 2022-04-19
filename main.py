import cv2
import numpy as np

class lineDetector:
    def __init__(self):
        self.frame = None
        self.cap = None
        self.gray_frame = None
        self.blurred_frame = None
        self.thresholded_frame = None
        self.canny_frame = None
        self.eroded_frame = None
        self.dilated_frame = None
        self.contour_list = []
        self.sorted_contour = []
        self.final_contours = []
        self.box = None
        self.points = []
        self.xmin = None
        self.ymin = None
        self.wmin = None
        self.hmin = None
        self.angle = None
        self.midx = None
        self.midy = None
        self.midpoint = None
        self.slope = None

    def capturing(self, thing):
        self.thing = thing
        self.cap = cv2.VideoCapture(self.thing)
        ret, self.frame = self.cap.read()
        self.frame = cv2.resize(self.frame, (640, 480), 0, 0, cv2.INTER_CUBIC)
        return self.frame

    def processor(self, frame = None):
        self.gray_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        self.blurred_frame = cv2.GaussianBlur(self.frame, (5, 5), 4)
        self.thresholded_frame = cv2.threshold(self.blurred_frame, cv2.THRESH_BINARY, 127, 255)
        self.canny_frame = cv2.Canny(self.thresholded_frame, 25, 75)
        self.eroded_frame = cv2.erode(self.canny_frame, (3, 3), 4)
        self.dilated_frame = cv2.dilate(self.eroded_frame, (3, 3))
        return self.dilated_frame

    def contours_processor(self, frame=None):
        self.contour_list, _ = cv2.findContours(self.dilated_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contour_list = list(self.contour_list)

        if len(self.contour_list) > 0:
            self.sorted_contour = sorted(self.contour_list, key=cv2.contourArea, reverse=True)
            self.final_contours = list(self.sorted_contour[0:2])

            for i in self.final_contours:
                m = cv2.moments(self.final_contours[i])
                if m["m00"] != 0:
                    c_x = int(m["m10"] / m["m00"])
                    c_y = int(m["m01"] / m["m00"])
                    self.points.append((c_x, c_y))
                self.box = cv2.minAreaRect(self.sorted_contour)
        return self.points, self.box

    def lux_aeterna(self):
        self.midx = (self.points[0][0] + self.points[1][0]) // 2
        self.midy = (self.points[1][0] + self.points[1][1]) // 2
        self.slope = (240 - self.midy) / (320 - self.midx)
        self.angle = np.arctan(self.slope)
        print(self.angle)
        return self.angle

def main():
    object_alpha = lineDetector()
    object_alpha.capturing(0)
    object_alpha.processor()
    object_alpha.contours_processor()
    object_alpha.lux_aeterna()

if __name__ == "main":
    main()
















