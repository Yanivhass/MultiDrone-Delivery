import cv2
import math
import numpy as np
import sys
import matplotlib as plt
def shape_detect(img):
        center, angle = (0, 0), 0
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)     
        ## mask of green (36,25,25) ~ (86, 255,255)
        # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
        mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

        ## slice the green
        imask = mask>0
        green = np.zeros_like(img, np.uint8)
        green[imask] = img[imask]
        
        _, threshold = cv2.threshold(green, 230, 255, cv2.THRESH_BINARY)  # from low to high convert to 255
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        triangle_center, pentagon_center = (0, 0), (0, 0)

        for cnt in contours:
            if 10000 > cv2.contourArea(cnt) > 100:
                approx = cv2.approxPolyDP(cnt, 0.06 * cv2.arcLength(cnt, True), True)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
                if len(approx) == 3:
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    triangle_center = (cx, cy)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
                    # cv2.putText(img, "Triangle", (cx, cy), font, 1, (255, 0, 0))
                # elif len(approx) == 4:
                #     cv2.putText(img, "Rectangle", (x, y), font, 1, (255, 0, 0))
                elif len(approx) == 5:
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    pentagon_center = (cx, cy)
                    cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
                    # cv2.putText(img, "Pentagon", (cx, cy), font, 1, (255))
                # elif 6 < len(approx) < 15:
                #     cv2.putText(img, "Ellipse", (x, y), font, 1, (255))
                # elif len(approx) != 4:
                #     cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
                #     cv2.putText(img, "Circle", (x, y), font, 1, (255))
        if triangle_center != (0, 0) and pentagon_center != (0, 0):
            if triangle_center[1] > pentagon_center[1]:
                up_or_down = 0  # down
            else:
                up_or_down = 1  # up
            # print(triangle_center, pentagon_center)
            if triangle_center[0] == pentagon_center[0]:
                if triangle_center[1] > pentagon_center[1]:
                    angle = -90
                else:
                    angle = 90
            elif triangle_center[1] == pentagon_center[1]:
                if triangle_center[0] > pentagon_center[0]:
                    angle = 0
                else:
                    angle = 180
            else:
                angle = math.degrees(math.atan(float(triangle_center[1] - pentagon_center[1]) /
                                               (triangle_center[0] - pentagon_center[0])))
                dy = triangle_center[1] - pentagon_center[1]
                dx = triangle_center[0] - pentagon_center[0]

                if dx > 0 and dy > 0:
                    angle = - angle
                elif dx > 0 and dy < 0:
                    angle = - angle
                elif dx < 0 and dy < 0:
                    angle = 180 - angle
                elif dx < 0 and dy > 0:
                    angle = abs(angle) - 180
            cv2.line(img, triangle_center, pentagon_center, (0, 255, 0), 2)
            line_x = int((triangle_center[0] + pentagon_center[0]) / 2)
            line_y = int((triangle_center[1] + pentagon_center[1]) / 2)
            center = (line_x, line_y)
            cv2.circle(img, center, 4, (0, 0, 255), -1)
            cv2.putText(img, 'ang: ' + str(round(angle)), (20, 30), 'font', 0.8, (0, 0, 255))

        return center, angle


''' Yaniv's notes:
Returns PID controller output for given variables
Input:
:param float list position - (x,y) values of drone location
:param float angle - drone's azimuth. [0-360).
:param float x1 - x value for current line start
:param float y1 - y value for current line start
:param float x2 - x value for current line end
:param float y2 - y value for current line end

note for future use:
if using A* paths with no straight lines, we could use
path[i-1],path[i+1] as approximation
'''

def image_broadcast():
        # ip = '192.168.1.195'
        # URL = 'http://' + ip + ':8080/shot.jpg'
        snapshot = 0
       
        # Create a VideoCapture object and read from input file
        # If the input is the camera, pass 0 instead of the video file name

        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        

        # Check if camera openeindex = 1 + cv::CAP_MSMF successfully +cv2.CAP_MSMF
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if ret == True:

                # Display the resulting frame
                cv2.imshow('Frame',frame)
                # shape_detect(frame)
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        cv2.destroyAllWindows()

# image_broadcast()