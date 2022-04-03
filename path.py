
import cv2
import urllib
import numpy as np
import math
import time
import sys, os, random, pygame
from math import sqrt, cos, sin, atan2
# import matplotlib.image as mpimg
from dijkstar import Graph, find_path
'''
class path_finder():

    def __init__(self):
        self.pub_vel = rospy.Publisher('bebop/cmd_vel', Twist, queue_size=1)
        self.pub_takeoff = rospy.Publisher('/bebop/takeoff', Empty, queue_size=1)
        self.pub_land = rospy.Publisher('/bebop/land', Empty, queue_size=1)
        self.battery_sub = rospy.Subscriber('/bebop/states/common/CommonState/BatteryStateChanged',
                                            CommonCommonStateBatteryStateChanged, self.battery)
        self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
        self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)
        self.battery_val = 0
        self.up_or_down = 0

        # Defining angle controller variebles-Ku=0.04 T=2. PID: p=0.024,i=0.024,d=0.006. PD: p=0.032, d=0.008. P: p=0.01
        self.kp_ang = 0.02  # 0.01
        self.ki_ang = 0
        self.kd_ang = 0
        self.ang_integral = 0
        self.ang_derivative = 0
        self.previous_ang = 0

        # Defining error controller variebles-Ku=0.14 T=6. PID: p=0.084,i=0.028,d=0.063. PD: p=0.112, d=1. P: p=0.07
        self.kp_err = 0.075
        self.ki_err = 0
        self.kd_err = 1.8
        self.err_integral = 0
        self.err_derivative = 0
        self.previous_err = 0

        self.ang_normal = 0.5
        self.err_normal = (1.0 / 800) * 16
        self.velocity = 0.0015

    def battery(self, data):
        self.battery_val = data.percent


'''

    '''
    Finds drone direction and location in an image
    
    '''
    def shape_detect(img):
        center, angle = (0, 0), 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)  # from low to high convert to 255
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            # if triangle_center[1] > pentagon_center[1]:
            #     self.up_or_down = 0  # down
            # else:
            #     self.up_or_down = 1  # up
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
            cv2.putText(img, 'ang: ' + str(round(angle)), (20, 30), font, 0.8, (0, 0, 255))

        return center, angle
'''
    def live_drone(self, img):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img = cv2.flip(img, 1)

        center = (250, 250)
        angle = 90.0
        center, angle = self.shape_detect(img)
        # x1, y1, x2, y2 = 0, 0, 1, 0
        font = cv2.FONT_HERSHEY_SIMPLEX

        global IS_FLYING
        if IS_FLYING == 0:
            global curr_line_indx
            curr_line_indx = 0
            print('num nodes: ', len(nodes_rrt_avg))
            print('current line: ', curr_line_indx)
            IS_FLYING = 1

        if self.dist(center, nodes_rrt_avg[curr_line_indx + 1].x_y_pos()) < RADIUS and curr_line_indx == len(
                nodes_rrt_avg) - 2:
            land = Empty()
            self.pub_land.publish(land)

        for i in range(len(nodes_rrt_avg) - 1):
            if self.dist(center, nodes_rrt_avg[i + 1].x_y_pos()) < RADIUS and curr_line_indx < len(
                    nodes_rrt_avg) - 2:
                curr_line_indx = i + 1
                # print('current line: ', curr_line_indx)

        if cv2.waitKey(1) == ord('s'):
            curr_line_indx = 0

        # if curr_line_indx + 1 < len(nodes_rrt_avg):
        (x1, y1) = nodes_rrt_avg[curr_line_indx].x_y_pos()
        (x2, y2) = nodes_rrt_avg[curr_line_indx + 1].x_y_pos()
        # global x
        # if x == 0:
        #     for i in range(len(nodes_rrt_avg) - 1):
        #         (x1, y1) = nodes_rrt_avg[i].x_y_pos()
        #         (x2, y2) = nodes_rrt_avg[i + 1].x_y_pos()
        #         print((x1, y1), (x2, y2))
        #         print(self.current_rrt_line_angle(x1, y1, x2, y2))
        #     x = 1

        cv2.imshow('image', img)

        cv2.waitKey(1)

        return center, angle, (x1, y1), (x2, y2)
'''
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
    def controllers(self, position, angle, x1, y1, x2, y2):
        # print('org angle: ', angle)
        current_ang = angle
        curr_line_x1 = x1
        curr_line_y1 = y1
        curr_line_x2 = x2
        curr_line_y2 = y2
        drone_x_location = position[0]
        drone_y_location = position[1]

        # ANGLE PID CONTROLLER-Finding the controller value
        self.ang_integral = self.ang_integral + current_ang
        self.ang_derivative = current_ang - self.previous_ang
        # curr_line_ang = 180 - math.degrees(math.atan((curr_line_y2 - curr_line_y1) / (curr_line_x2 - curr_line_x1)))
        curr_line_ang = self.current_rrt_line_angle(curr_line_x1, curr_line_y1, curr_line_x2, curr_line_y2)
        # print(curr_line_ang)
        ang_controller_value = round(
            (self.kp_ang * (curr_line_ang - current_ang) * self.ang_normal + self.ki_ang * self.ang_integral +
             self.kd_ang * self.ang_derivative), 2)
        # print('curr_line_ang: ', curr_line_ang, '\n', 'current_ang: ', current_ang, 'diff: ', curr_line_ang -
        #       current_ang)
        self.previous_ang = current_ang

        # ERROR PID CONTROLLER-Finding the controller value
        if curr_line_x1 == curr_line_x2:
            rrt_curr_line_a = 0
            rrt_curr_line_b = 0
        else:
            rrt_curr_line_a = float(curr_line_y2 - curr_line_y1) / (curr_line_x2 - curr_line_x1)
            rrt_curr_line_b = curr_line_y1 - rrt_curr_line_a * curr_line_x1
        #    normalization_coeff=cv_image.shape[1] / 2#****
        # Distance between poine and  line formula
        # current_err = ((abs(-1 * rrt_curr_line_a * drone_x_location + drone_y_location - rrt_curr_line_b)) / (
        #     math.sqrt(rrt_curr_line_a ** 2 + 1)))  # /(normalization_coeff)
        current_err, _ = self.dist_line_to_point((x1, y1), (x2, y2), position)
        # print('curr err: ', current_err)
        current_err = current_err * self.err_normal
        self.err_integral = self.err_integral + current_err
        self.err_derivative = current_err - self.previous_err
        global right, left
        right, left = self.right_or_left(curr_line_x1, rrt_curr_line_a, rrt_curr_line_b, drone_x_location, drone_y_location)
        err_controller_value = 0
        err = Int16()

        if right == 1 and left == 0:
            err_controller_value = round(
                -1 * (self.kp_err * current_err + self.ki_err * self.err_integral + self.kd_err * self.err_derivative), 4)
            err.data = -1 * current_err / self.err_normal
        elif right == 0 and left == 1:
            err_controller_value = round \
                ((self.kp_err * current_err + self.ki_err * self.err_integral + self.kd_err * self.err_derivative),
                                         4)
            err.data = current_err / self.err_normal

        self.previous_err = current_err
        # print('P: ', self.kp_err * current_err, 'D: ', self.kd_err * self.err_derivative)
        ang = Int16()
        ang.data = curr_line_ang - current_ang
        # print(curr_line_ang, current_ang, ang.data)
        self.pub_angle.publish(ang)

        self.pub_error.publish(err)
        return err_controller_value, ang_controller_value

    def image_broadcast(self):
        # ip = '192.168.1.195'
        # URL = 'http://' + ip + ':8080/shot.jpg'
        snapshot = 0
        cap = cv2.VideoCapture(1)
        while True:
            try:
                # img_arr = np.array(bytearray(urllib.urlopen(URL).read()), dtype=np.uint8)
                _, img = cap.read()
            except Exception as e:
                print(e)
                return

            # img = cv2.imdecode(img_arr, -1)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img = cv2.flip(img, 1)
            if snapshot == 0:
                snapshot = 1

                global YDIM, XDIM, WINSIZE
                YDIM, XDIM, _ = img.shape
                print('X: ', YDIM, ', ', 'Y: ', XDIM)
                WINSIZE = [YDIM, XDIM]
                imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                global thresh
                _, thresh = cv2.threshold(imgray, 100, 255, 0)
                # print(type(thresh))
                kernel = np.ones((3, 3), np.uint8)
                thresh = cv2.erode(thresh, kernel, iterations=5)
                thresh = cv2.dilate(thresh, kernel, iterations=9)

                _, contours, _ = cv2.findContours(~thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                boxs = []
                padding = 45
                print('contours: ', len(contours))
                for item in contours:
                    # rect = cv2.minAreaRect(item)
                    # if cv2.contourArea(item) > 300:
                    (x, y), (w, h), angle = cv2.minAreaRect(item)
                    rect = ((x, y), (w + padding, h + padding), angle)
                    # print('rect: ', x, y, w, h, angle)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # print('box: ', box)
                    boxs.append(box)

                for box in boxs:
                    # cv2.fillPoly(thresh, [box], color=(0, 0, 0))
                    cv2.drawContours(thresh, [box], 0, (0, 0, 0), -1)

                self.rrt()
                bg = pygame.image.load('rrt_bg.png').convert()
                arrow_org = pygame.image.load('arrow.png')  # .convert_alpha()
                arrow_org = pygame.transform.scale(arrow_org, (50, 50))
                font = pygame.font.Font('freesansbold.ttf', 16)
            else:
                screen.blit(bg, [0, 0])
                pos, ang, (x1, y1), (x2, y2) = self.live_drone(img)
                pygame.draw.line(screen, red, (x1, y1), (x2, y2), 3)
                err_corr, ang_corr = self.controllers(pos, ang, x1, y1, x2, y2)

                # screen.blit(font.render('err_corr: ' + str(err_corr), True, (255, 0, 0)), [10, 80])
                # screen.blit(font.render('ang_corr: ' + str(ang_corr), True, (255, 0, 0)), [10, 110])

                arrow = pygame.transform.rotate(arrow_org, ang)
                screen.blit(arrow, [pos[0] - 30, pos[1] - 30])

                text_pos = font.render('pos: ' + str(pos), True, (255, 0, 0))
                screen.blit(text_pos, [10, 10])

                text_ang = font.render('ang: ' + str(round(ang, 2)), True, (255, 0, 0))
                screen.blit(text_ang, [10, 30])

                line_ang = str(round(self.current_rrt_line_angle(x1, y1, x2, y2), 2))
                text_line_ang = font.render('Current line angle: ' + line_ang, True, (255, 0, 0))
                screen.blit(text_line_ang, [10, 50])
                # if self.up_or_down:
                #     text_dir = font.render('Drone dir: UP', True, (255, 0, 0))
                # else:
                #     text_dir = font.render('Drone dir: DOWN', True, (255, 0, 0))
                # screen.blit(text_dir, [10, 30])

                # if right:
                #     text_side = font.render('Path side: RIGHT', True, (255, 0, 0))
                # else:
                #     text_side = font.render('Path side: LEFT', True, (255, 0, 0))
                # screen.blit(text_side, [10, 50])

                dot = pygame.draw.circle(screen, (255, 0, 255), pos, 8)
                batt = font.render('Battery: ' + str(self.battery_val), True, (0, 0, 255))
                screen.blit(batt, [10, 75])

                pygame.display.update()

                # print('error_correction: {} \nangle_correction: {}\n-------------------------------'
                # .format(err_corr, ang_corr))
                self.send_cmd(err_corr, ang_corr)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit(0)

    def dist(self, p1, p2):
        return sqrt((p1[0 ] -p2[0] ) *(p1[0 ] -p2[0] ) +(p1[1 ] -p2[1] ) *(p1[1 ] -p2[1]))

    def dist_line_to_point(self, start, end, point):
        if end[0] == start[0]:
            x_cross = start[0]
            y_cross = point[1]
            current_err = abs(point[0] - x_cross)
        elif end[1] == start[1]:
            x_cross = point[0]
            y_cross = start[1]
            current_err = abs(point[1] - y_cross)
        else:
            rrt_curr_line_a = float(end[1] - start[1]) / (end[0] - start[0])
            rrt_curr_line_b = start[1] - rrt_curr_line_a * start[0]

            x_cross = (rrt_curr_line_a * (point[1] - rrt_curr_line_b) + point[0]) / (1 + rrt_curr_line_a ** 2)
            y_cross = rrt_curr_line_a * x_cross + rrt_curr_line_b
            #    normalization_coeff=cv_image.shape[1] / 2#****
            # Distance between poine and  line formula
            current_err = (float(abs(-1 * rrt_curr_line_a * point[0] + point[1] - rrt_curr_line_b)) /
                (math.sqrt(rrt_curr_line_a ** 2 + 1)))
        # check if the drone is in the area of the current line

        in_x_range = False
        in_y_range = False
        if end[0] < start[0]:
            if end[0] < x_cross < start[0]:
                in_x_range = True
        if end[0] > start[0]:
            if end[0] > x_cross > start[0]:
                in_x_range = True

        if end[1] < start[1]:
            if end[1] < y_cross < start[1]:
                in_y_range = True
        if end[1] > start[1]:
            if end[1] > y_cross > start[1]:
                in_y_range = True
        is_cross = in_x_range and in_y_range
        return current_err, is_cross


    # The following function gets two points (p1-x_y position of nearst node,p2-x_y position of random node) and
    # reutrns x_y position of a point with epsilon distance from p1 and equal azimuth to the azimuth between p1 and p2
    # Yaniv's notes: I have no idea what function they are talking about
    ''' Yaniv's notes:
    line equation is defined as y = line_a*x + line_b
    for given x_location, y_location returns if drone is above or below the line
    '''
    def right_or_left(self, line_x, line_a, line_b, x_location, y_location):
        right = 0
        left = 0
        if self.up_or_down == 1:  # drone is up
            if line_a == 0 and line_b == 0:
                if x_location >= line_x:
                    left = 1
                else:
                    right = 1
            elif y_location > line_a * x_location + line_b:
                right = 1
            elif y_location <= line_a * x_location + line_b:
                left = 1
        if self.up_or_down == 0:  # drone is down
            if line_a == 0 and line_b == 0:
                if x_location >= line_x:
                    right = 1
                else:
                    left = 1
            elif y_location > line_a * x_location + line_b:
                left = 1
            elif y_location <= line_a * x_location + line_b:
                right = 1
        return right, left

    # get a line's azimuth
    def current_rrt_line_angle(self, x_start, y_start, x_end, y_end):
        curr_line_ang = 0
        if x_start == x_end:
            if y_start <= y_end:
                curr_line_ang = -90
            else:
                curr_line_ang = 90
        elif y_start == y_end:
            if x_start <= x_end:
                curr_line_ang = 0
            else:
                curr_line_ang = 180
        elif y_end <= y_start:  # line direction is up
            if x_end <= x_start:
                curr_line_ang = 180 - math.degrees(math.atan((float(y_end - y_start) / (x_end - x_start))))
            else:
                curr_line_ang = -1 * math.degrees(math.atan((float(y_end - y_start) / (x_end - x_start))))
        elif y_end > y_start:  # line direction is down
            if x_end <= x_start:
                curr_line_ang = -1 * (180 + math.degrees(math.atan((float(y_end - y_start) / (x_end - x_start)))))
            else:
                curr_line_ang = -1 * math.degrees(math.atan((float(y_end - y_start) / (x_end - x_start))))

        return curr_line_ang

    def step_from_to(self, p1, p2):
        if self.dist(p1, p2) < EPSILON:
            return p2
        else:
            theta = atan2(p2[1] - p1[1], p2[0] - p1[0])
            return p1[0] + EPSILON * cos(theta), p1[1] + EPSILON * sin(theta)

    def find_b(self, az, obs_x_center, obs_y_center):
        return obs_y_center - (math.tan(math.radians(az)) * obs_x_center)

    # find intersection point of two lines
    def find_cutting_point(self, a1, a2, b1, b2):
        x = round((b2 - b1) / (a1 - a2))
        y = round(a1 * x + b1)
        return x, y

    def send_cmd(self, err_corr, ang_corr):
        twist = Twist()
        twist.linear.x = self.velocity  # need to convert from px to distance
        twist.linear.y = err_corr
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = ang_corr
        self.pub_vel.publish(twist)

    def rrt(self):
        # rospy.init_node('path_finder', anonymous=True)
        # ros = ros()

        # must be done in order to use pygame
        pygame.init()

        # width and highth of game
        global screen
        screen = pygame.display.set_mode(WINSIZE)

        # title of the game
        pygame.display.set_caption('RRT Drone')
        # global thresh

        map = pygame.surfarray.make_surface(thresh)
        screen.blit(map, [0, 0])

        # convert the screen to 2d array,each cell is a pixel.can be usefull

        # screenarray = pygame.PixelArray(screen)

        # update the screen

        # pygame.display.flip()

        # Drawing start and end point and middle of obstacles

        pygame.draw.circle(screen, blue, source, int(EPSILON))
        pygame.draw.circle(screen, red, goal, int(EPSILON))
        pygame.display.update()

        # Initialization of nodes-vector of node objects
        node_name = 1
        nodes = []
        global nodes_rrt_avg
        nodes_rrt_avg = []
        # defining the source as start node

        start_node = node(source[0], source[1], node_name)
        nodes.append(start_node)
        path_found = 0

        if (map.get_at(source)[0:3] == (0, 0, 0)) or (map.get_at(goal)[0:3] == (0, 0, 0)):
            print('start/end point on an obstacle')
            sys.exit()
        else:
            for i in range(NUMNODES):
                node_name = node_name + 1
                # pick a random node

                rand = random.randint(0, YDIM - 1), random.randint(0, XDIM - 1)
                # print(YDIM, XDIM)
                # checks for valid position - avoiding obstacels

                while map.get_at(rand)[0:3] != (255, 255, 255):  # if the random point is on obstacle
                    rand = random.randint(0, YDIM - 1), random.randint(0, XDIM - 1)  # choose new random point
                # nearest node
                nn = nodes[0]
                # search for the nearests node to the random node (RRT*)
                for p in nodes:
                    if self.dist(p.x_y_pos(), rand) < self.dist(nn.x_y_pos(), rand):
                        nn = p
                x_pos, y_pos = self.step_from_to(nn.x_y_pos(), rand)
                x_pos = round(x_pos)
                y_pos = round(y_pos)
                newnode = node(x_pos, y_pos, node_name)
                # checks if the new node is valid
                while map.get_at((int(round(newnode.x_pos)), int(round(newnode.y_pos))))[0:3] != (
                255, 255, 255):  # if the newnode is on obstacle
                    rand = random.randint(0, YDIM - 1), random.randint(0, XDIM - 1)
                    while map.get_at(rand)[0:3] != (255, 255, 255):
                        rand = random.randint(0, YDIM - 1), random.randint(0, XDIM - 1)
                    nn = nodes[0]
                    for p in nodes:
                        if self.dist(p.x_y_pos(), rand) < self.dist(nn.x_y_pos(), rand):
                            nn = p
                    x_pos, y_pos = self.step_from_to(nn.x_y_pos(), rand)
                    x_pos = round(x_pos)
                    y_pos = round(y_pos)
                    newnode = node(x_pos, y_pos, node_name)
                nodes.append(newnode)
                graph.add_edge(nn.node_num, newnode.node_num, 1)
                pygame.draw.line(screen, blue, nn.x_y_pos(), newnode.x_y_pos())
                pygame.display.update()
                if self.dist(newnode.x_y_pos(), goal) <= EPSILON:
                    path_found = 1
                    path = find_path(graph, 1,
                                     node_name)  # type : object ##Finding shortest path from all the nodes of rrt algorithm using dijkstra
                    # print(len(path.nodes), 'Number of nodes that the path includes')
                    # print(newnode.x_y_pos(),'Found Path')
                    # print(node_name,'num of nodes')
                    # print(len(nodes),'list of nodes size')
                    break
            azimuths = []

            # Drawing shortest path from all the nodes of rrt algorithm using dijkstra
            if (path_found == 0):
                print('No path found!')
                sys.exit()
            else:
                for i in range(len(path.nodes) - 1):
                    pygame.draw.line(screen, red, nodes[path.nodes[i] - 1].x_y_pos(),
                                     nodes[path.nodes[i + 1] - 1].x_y_pos())
                    azimuth = atan2(nodes[path.nodes[i + 1] - 1].y_pos - nodes[path.nodes[i] - 1].y_pos,
                                    nodes[path.nodes[i + 1] - 1].x_pos - nodes[path.nodes[i] - 1].x_pos)
                    azimuths.append(azimuth)
                    # print('node num',path.nodes[i], 'to node num', path.nodes[i + 1])
                    # print('x_y position',nodes[path.nodes[i]-1].x_y_pos(),'to x_y position',nodes[path.nodes[i+1]-1].x_y_pos(),'azimuth',azimuth)
                pygame.display.update()

                # drawing rrt_avg path and saving all the azimuths (rrt_avg_azimuths) and distances (rrt_avg_distances) in rrt_avg path
                global rrt_avg_azimuths
                rrt_avg_azimuths = []
                rrt_avg_distances = []
                for j in range(len(path.nodes)):
                    if j % RRT_AVG == 0 and j != len(path.nodes):
                        pygame.draw.circle(screen, black,
                                           [int(nodes[path.nodes[j] - 1].x_pos), int(nodes[path.nodes[j] - 1].y_pos)],
                                           int(3))
                        nodes_rrt_avg.append(nodes[path.nodes[j] - 1])
                    if j == len(path.nodes):
                        pygame.draw.circle(screen, black,
                                           [int(nodes[path.nodes[j] - 1].x_pos), int(nodes[path.nodes[j] - 1].y_pos)],
                                           int(3))
                        nodes_rrt_avg.append(nodes[path.nodes[j] - 1])
                    pygame.display.update()
                final_node_rrt_avg = node(goal[0], goal[1], node_name + 1)
                pygame.draw.circle(screen, black, [int(final_node_rrt_avg.x_pos), int(final_node_rrt_avg.y_pos)],
                                   int(3))
                nodes_rrt_avg.append(final_node_rrt_avg)
                pygame.display.update()

                # num_of_points = Int16()
                # num_of_points.data = len(nodes_rrt_avg)
                # self.pub_path.publish(num_of_points)

                for k in range(len(nodes_rrt_avg) - 1):
                    rrt_avg_distances.append(
                        round(self.dist(nodes_rrt_avg[k].x_y_pos(), nodes_rrt_avg[k + 1].x_y_pos()), 2))
                    delta_x = abs(nodes_rrt_avg[k].x_pos - nodes_rrt_avg[k + 1].x_pos)
                    delta_y = nodes_rrt_avg[k].y_pos - nodes_rrt_avg[k + 1].y_pos
                    if delta_x == 0 and delta_y > 0:
                        rrt_avg_azimuths.append(90)
                    elif delta_x == 0 and delta_y < 0:
                        rrt_avg_azimuths.append(-90)
                    else:
                        rrt_avg_azimuths.append(round(math.degrees(np.arctan(delta_y / delta_x)), 2))

                    # print('from: ', nodes_rrt_avg[k].x_y_pos(), 'to: ', nodes_rrt_avg[k+1].x_y_pos(), 'azimuth: ', rrt_avg_azimuths[k], '[deg]', 'distance: ', rrt_avg_distances[k], '[px]')
                    pygame.draw.line(screen, green, nodes_rrt_avg[k].x_y_pos(), nodes_rrt_avg[k + 1].x_y_pos())
                    pygame.draw.circle(screen, green,
                                       (int(nodes_rrt_avg[k + 1].x_y_pos()[0]), int(nodes_rrt_avg[k + 1].x_y_pos()[1])),
                                       RADIUS, 2)
                    pygame.display.update()
                pygame.image.save(screen, 'rrt_bg.png')

                # self.send_cmd(rrt_avg_distances, rrt_avg_azimuths)
        return

