import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


class DetectLaneLine :

    def __init__(self, input_img) -> None:
        self.input_img = input_img
        self.height = input_img.shape[0]
        self.width = input_img.shape[1]


    def blur_image (self):
        gray = cv2.cvtColor(self.input_img, cv2.COLOR_RGB2GRAY)
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        return blur_gray

    def edge_detection (self): 
        blur_gray = self.blur_image()
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        return edges


    def generate_mask (self):
        edges = self.edge_detection()
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(50, self.height), 
                                (self.width -50, self.height), 
                                (self.width -250, self.height//2), 
                                (250, self.height//2)]], dtype = np.int32)
        cv2.fillPoly(mask, roi_vertices, 255) 
        masked_edges = cv2.bitwise_and(edges, mask)
        cv2.imwrite('masked.jpg', masked_edges)
        return masked_edges
    
    def HoughTransform (self):
        masked_edges = self.generate_mask ()
        edges = self.edge_detection()
        rho = 1
        theta = np.pi/180
        threshold = 15
        min_line_length = 10
        max_line_gap = 1

        line_image= np.zeros_like(self.input_img)
        # Run Hough on edge detected image
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                    min_line_length, max_line_gap)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)

        color_edges = np.dstack((edges, edges, edges))
        combo = cv2.addWeighted(self.input_img, 0.8, line_image, 1, 0)
        return combo


def preprocess_video(input):
    output_file= 'result.mp4'    
    cap = cv2.VideoCapture(input)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    frame_number = 1
    print("[INFO] System Process Initialized... ")

    while True:     
        ret, frame = cap.read()
        if not ret:
            break
        det_lane = DetectLaneLine(frame)
        pro_img = det_lane.HoughTransform()
        vid_writer.write(pro_img)
        cv2.imshow('frame', pro_img)
        if cv2.waitKey(1)==ord('q'):
                    break
    cv2.destroyAllWindows()
    vid_writer.release()



if __name__ == "__main__":
     preprocess_video('car.mp4')
     












