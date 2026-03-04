import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time as t
import pyautogui as mouse_control

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

MODEL_PATH = "hand_landmarker.task"

mouse_control.FAILSAFE = True
mouse_control.PAUSE = 0 
mouse_control.MINIMUM_DURATION = 0

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH) #width
cap.set(cv.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT) #height

if not cap.isOpened():
  print("Cannot open camera")
  exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv.flip(frame, 1) #mirror
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #give camera color
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    finger_pos = None
    rx,ry = 954, 540
        
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    if result.hand_landmarks:
        # Get Index Finger Tip (Landmark 8)
        landmark = result.hand_landmarks[0][8] 
        finger_pos = (int(landmark.x * SCREEN_WIDTH), int(landmark.y * SCREEN_HEIGHT))
        cv.circle(frame, finger_pos, 20, (0, 0, 255), -1)
        print(f"|X {finger_pos[0]}| Y {finger_pos[1]}|")
        x = int(finger_pos[0])
        y = int(finger_pos[1]) 
        lx = x - rx
        ly = y - ry
        #mouse_control.moveRel(lx * 0.2,ly * 0.2)
        mouse_control.moveTo(x,y)
    cv.imshow('Mouse Hand Tracker', frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
    