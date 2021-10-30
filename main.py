
from matplotlib.pyplot import plot

from djitellopy import Tello
from djitellopy import TelloSwarm
import cv2
# import pygame
import time
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import colorama
from colorama import Fore
from colorama import Style
import argparse

import detect
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

colorama.init()




#cap = cv.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
#### LOAD MODEL
## tello Names
'''
classesFile = r"/home/user_2/Downloads/tello_origin/tello.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
'''
# Load YOLOv5 Drone detection model
device = select_device('cpu') # cuda device, i.e. 0 or 0,1,2,3 or cpu
weights = r"weights.pt"
model = attempt_load(weights, map_location=device)  # load FP32 model
# modelc = load_classifier(name='resnet50', n=2)  # initialize
# modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()



#GLOBAL PARAMS
TIMER = 0
TIME_STEP = 0.001
SETPOINT = 10
SIM_TIME = 100
INITIAL_X = 0
INITIAL_Y = -100

def predict(img,
            conf_thres=0.25,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            view_img=True,
            imgsz=640,
            agnostic_nms=False,  # class-agnostic NMS
            ):
    stride = int(model.stride.max())  # model stride
    dataset = LoadImages(img, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        img = (img/255.0).float() # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
             img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img,augment=True)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        classify = False
        # Apply Classifier
        if classify:
            im0s = ''
            # pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt


def findObjects(outputs, img):
    global h, w, y, x
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    faces = []

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        # cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
        #            (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # faces.append([x,y,w,h,classNames[classIds[i]]])

    return faces



# Face recognition cascade
# from yolo import net, cap

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Speed of the drone
S = 30
found_first = False

'''
class FrontEnd(object):

    def __init__(self):
        # Init pygame
        # pygame.init()

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10
        self.faceFound = False
        self.send_rc_control = True


    def run(self):
        global found_first
        poses = np.array([])
        times = np.array([])
        timer = 0

        if not self.tello.connect():
            print("Tello not connected")
            return

        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        # In case streaming is on. This happens when we quit this program without the escape key.
        if not self.tello.streamoff():
            print("Could not stop video stream")
            return

        if not self.tello.streamon():
            print("Could not start video stream")
            return

        frame_read = self.tello.get_frame_read()
        if self.tello.takeoff():
            print('Takeoff successfull')
        self.tello.connect()
        time.sleep(1)
        self.tello.move_up(100)
        star = time.time()
        # yaw_pid = PID(KP_yaw,KI_yaw,KD_yaw)
        # updown_pid = PID(KP_updown,KI_updown,KD_updown)
        # face_pid = PID(KP_face,KI_face,KD_face)
        ts = 0

        while True:


            # pygame.event.pump()
            # frame = frame_read.frame
            # blob = cv.dnn.blobFromImage(frame, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
            # net.setInput(blob)
            layersNames = net.getLayerNames()
            outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
            outputs = net.forward(outputNames)
            faces = findObjects(outputs,frame)
            ####################################################
            if time.time() - star > 90:
                break

            for (x, y, w, h,object_name) in faces:
                self.faceFound = True
                cx = (2*x+w)/2
                cy = (2*y+h)/2
                faceSize = (w+h)/2
                timer += 1
            if self.faceFound:
                if object_name == 'dustPan' :
                    print(Fore.YELLOW + "++++++WARNING OBSTACLE HAS DETECTED++++++" + Style.RESET_ALL)
                    yaw_dif = abs(cx - 480)
                    updo_dif = abs(cy - 300)
                    face_dif = abs(faceSize - 140)
                    if face_dif == 0:
                        face_dif = 1
                    if yaw_dif == 0:
                        yaw_dif = 1
                    if updo_dif == 0:
                        updo_dif = 1
                    yaw_velocity = 0
                    self.yaw_velocity = yaw_velocity

                    updown_velocity = updown_pid.compute(cy, 350)
                    updown_velocity = np.clip(updown_velocity, -30, 30)
                    self.up_down_velocity = updown_velocity

                    face_velocity = face_pid.compute(faceSize, 100)
                    face_velocity = np.clip(face_velocity, -30, 30)
                    self.for_back_velocity = face_velocity
                    poses = np.append(poses, yaw_velocity)
                    times = np.append(times, timer)

                elif object_name == 'mohammad.zoabi':
                    print(Fore.GREEN + "++++++TRACKING MOHAMMED++++++" + Style.RESET_ALL)
                    found_first = True
                    yaw_dif = abs(cx - 480)
                    updo_dif = abs(cy - 300)
                    face_dif = abs(faceSize - 140)
                    if face_dif == 0:
                        face_dif = 1
                    if yaw_dif == 0:
                        yaw_dif = 1
                    if updo_dif == 0:
                        updo_dif = 1
                    yaw_velocity = yaw_pid.compute(480, cx)
                    yaw_velocity = np.clip(yaw_velocity, -40, 40)
                    self.yaw_velocity = yaw_velocity

                    updown_velocity = updown_pid.compute(cy, 180)
                    updown_velocity = np.clip(updown_velocity, -30, 30)
                    self.up_down_velocity = updown_velocity

                    face_velocity = face_pid.compute(faceSize, 140)
                    face_velocity = np.clip(face_velocity, -30, 30)
                    self.for_back_velocity = face_velocity
                    poses = np.append(poses, yaw_velocity)
                    times = np.append(times, timer)



            elif not found_first:
                print(Fore.BLUE + Style.BRIGHT + "++++++LOOKING FOR MOHAMMED++++++" + Style.RESET_ALL)
                self.for_back_velocity = 0
                self.yaw_velocity = 40
                self.up_down_velocity = 0
            else:
                self.for_back_velocity = 0
                self.yaw_velocity = 0
                self.up_down_velocity = 0
            self.update()
            cv2.imshow('img', frame)
            frame_read.out.write(frame)
            self.faceFound = False
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


        self.tello.end()
        print ("we get release\n")
        frame_read.out.release()
        frame_read.cap.release()
        cv2.destroyAllWindows()
        return 0


    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

'''
if __name__ == "__main__":

    device = select_device("cpu")
    weights = 'weights.pt'
    imgsz = 640

    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names

    swarm = TelloSwarm.fromIps([
    #      # "192.168.144.88",
    #     "192.168.50.151"
    ])

    swarm.connect()
    # swarm.streamon()
    for tello in swarm:
        print(tello.get_battery())
        # tello.takeoff()
        # tello.land()
        tello.streamon()
        frame_read = tello.get_frame_read() #720x960x3
        # cv2.imwrite("picture.png", frame_read.frame)
        # i = i + 1
    # frame_read = swarm.get_frame_read()

    # img = frame_read.frame

    '''
    bounding_box
    [0,1] - (x,y) of top left corner
    [2,3] - (x,y) of bottom right corner
    '''
    # bounding_box = detect.run(weights='weights.pt',source='picture2.png',nosave=False)
    # img = cv2.imread("picture.png")
    # predict(img)

    # frontend = FrontEnd()

    # run frontend
    # frontend.run()
    # opt = parse_opt()
    # main(opt)

    while True:

        # img = swarm[0].get_frame_read().frame  # 720x960x3
        bounding_box = detect.run(weights='weights.pt', source='picture2.png', nosave=False)
        img0 = cv2.imread("picture2.png")
        img = cv2.rotate(img0, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(img, (480,640), interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ', img.shape)
        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        img = img.permute(2,1,0).unsqueeze(0)
        # img = img.reshape((3, imgsz,imgsz)).unsqueeze(0)
        # image dims should be (1,3,480,640)

        t1 = time_synchronized()
        pred = model(img)[0]
        pred = non_max_suppression(pred)
        t2 = time_synchronized()



        for i, det in enumerate(pred):
            # p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    print(*xyxy)








