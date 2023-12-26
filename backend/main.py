from App import App
import cv2
import numpy as np

app = App()

cam1 = '/home/harito/Videos/Cam1.mp4'
cam2 = '/home/harito/Videos/Cam2.mp4'

cap1 = cv2.VideoCapture(cam1)
cap2 = cv2.VideoCapture(cam2)

cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)

i = 0
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break
    
    real_frame = cv2.hconcat([frame1, frame2])
    if i % 24 == 0:
        # Detect
        frame1_detect = app.detect_objects(frame1)
        frame2_detect = app.detect_objects(frame2)
        frame1_detect = cv2.cvtColor(np.array(app.draw_picture_detect(frame1_detect[1])), cv2.COLOR_RGB2BGR)
        frame2_detect = cv2.cvtColor(np.array(app.draw_picture_detect(frame2_detect[1])), cv2.COLOR_RGB2BGR)
        detect_frame = cv2.hconcat([frame1_detect, frame2_detect])        
    
    if i % 240 == 0:
        # Face recognition
        
        pass
    
    cv2.imshow('Cam', cv2.vconcat([real_frame, detect_frame]))
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
