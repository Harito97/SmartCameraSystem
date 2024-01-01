from App import App
import cv2
import numpy as np

app = App()

cam1 = '/home/harito/Videos/Cam1.mp4'
cam2 = '/home/harito/Videos/Cam2.mp4'

cap1 = cv2.VideoCapture(cam1)
cap2 = cv2.VideoCapture(cam2)

cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
# #######################################
# Kích thước của ảnh
height = 1440
width = 640
# Tạo một hình ảnh màu đen
black_image = np.zeros((height, width, 3), dtype=np.uint8)
terminal_frame = np.zeros((1440, 2560, 3), dtype=np.uint8)

# ########################################################
# Hàm để thay đổi kích thước ảnh để chúng có kích thước giống nhau
def resize_image(image, target_height=1440, target_width=640):
    resized_img = cv2.resize(image, (target_width, target_height))
    return resized_img

i = 0
list_frame = [black_image] * 8
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break
    
    real_frame = cv2.hconcat([frame1, frame2])

    if i % 24 == 0:
        # Detect
        result1_detect = app.detect_objects(frame1)
        result2_detect = app.detect_objects(frame2)
        frame1_detect = cv2.cvtColor(np.array(app.draw_picture_detect(result1_detect[1])), cv2.COLOR_RGB2BGR)
        frame2_detect = cv2.cvtColor(np.array(app.draw_picture_detect(result2_detect[1])), cv2.COLOR_RGB2BGR)
        detect_frame = cv2.hconcat([frame1_detect, frame2_detect])        
    
        if i % 10 == 0:
            # ReID & Face recognition
            face_in_room = []
            for person1 in result1_detect[0]:
                person1 = resize_image(person1)
                person = [person1]
                for person2 in result2_detect[0]:
                    if app.same_object(person1, person2):
                        person.append(resize_image(person2))
                for per in person:
                    # print(per)
                    data = app.face_similar(per)
                    if data[0]:
                        per = data[2]
                        face_in_room.append((data[1], person, per))
                        break
                    else:
                        face_in_room.append(('Unknown', person))
            
            end_point = len(face_in_room) if len(face_in_room) < 8 else 8
            for j in range(end_point):
                # list_frame[i] = resize_image(face_in_room[i][2]) if len(face_in_room[i]) == 3 else resize_image(face_in_room[i][1][0])
                list_frame[j] = face_in_room[j][2] if len(face_in_room[j]) == 3 else face_in_room[j][1][0]
           
            reid_facerecog_frame = cv2.hconcat(list_frame)        
    
    cv2.imshow('Cam', cv2.vconcat([cv2.hconcat([cv2.vconcat([real_frame, detect_frame]), terminal_frame]), reid_facerecog_frame]))
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
