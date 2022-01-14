import cv2
from djitellopy import Tello
from djitellopy import TelloSwarm
''' to connect a new drone use this
tello=Tello()
tello.connect()
tello.connect_to_wifi(ssid='AndroidAP', password='seru5625')
'''
swarm = TelloSwarm.fromIps([
     #"192.168.127.88",
     "192.168.127.151"
    ])
i= 0
swarm.connect()
for tello in swarm:
    while(True):
        print(tello.get_battery())
        # tello.takeoff()
        # tello.rotate_clockwise(90) #rotate to face north
        tello.streamon()
        frame_read = tello.get_frame_read() #720x960x3
        cv2.imwrite(f"drone_pics\\picture{i}.png", frame_read.frame)
        cv2.waitKey(1)
        i = i + 1
    # frame_read = swarm.get_frame_read()

    # img = frame_read.frame
