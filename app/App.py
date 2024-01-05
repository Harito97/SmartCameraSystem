from ultralytics import YOLO
from ultralytics.engine.results import Results
import face_recognition
import pickle
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

class List:
    def __init__(self):
        self.data = []
    
    def add(self, ele):
        if len(self.data) <= 4:
            self.data.insert(0, ele)
        else:
            self.data = [ele] + self.data[0:3]

class App:
    def __init__(self):
        self.yolo_model = YOLO('yolov8n-seg.pt')
        print("Setting YOLO model for detect ...")
        self.track1_history = defaultdict(lambda: [])
        self.track2_history = defaultdict(lambda: [])
        print('Setting track history ...')
        self.face_db_regis = pickle.loads(open("data/face_dictionary_extra.pkl", "rb").read())
        print("Set face recognition model with face_recognition \nSetup done!")

    # Detect and Track objects 
    def detect_track_objects(self, image:np.ndarray, cam:int):
        results = self.yolo_model.track(source=image, persist=False, classes=0)
        # Get the boxes and track IDs
        if results[0].boxes.id == None:
            return None, None, None
        
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        bounding_boxes = []
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            if cam == 1:
                track = self.track1_history[track_id]
            else:
                track = self.track2_history[track_id]

            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 15: 
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 200, 0), thickness=10)

            # Get bounding box and it's id 
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)
            bounding_boxes.append((image[ymin:ymax, xmin:xmax, :], track_id))

        return sorted(bounding_boxes, key=lambda x: x[1]), annotated_frame, track_ids

    # Object matching
    @staticmethod
    def load_reid_dict():
        reid_dict = {}
        loaded_data = np.load('data/histograms.npz')
        for key in loaded_data.files:
            data = List()
            data.add(loaded_data[key][0])
            data.add(loaded_data[key][1])
            reid_dict[key] = data
        return reid_dict 
    
    # Face recognition from each object crop and id by reid model 
    def __face_rec(self, image:np.ndarray):
        """
        Note: image input should only have a object person 
        face_recognition_model: model be used (this case is import face_recognition)
        Para:
            image: numpy array it can be the PIL.Image with convert in RGB mode
            ***To know more about the para then see this function in library***
            def load_image_file(file, mode='RGB'):
                ---
                Loads an image file (.jpg, .png, etc) into a numpy array

                :param file: image file name or file object to load
                :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white) are supported.
                :return: image contents as numpy array
                ---
                im = PIL.Image.open(file)
                if mode:
                    im = im.convert(mode)
                return np.array(im)
        Result:
            (bounding box of face, feature vector of the face)
        """
        bounding_boxes = face_recognition.face_locations(image)
        if len(bounding_boxes) < 1:
            return None
        feature_vectors = face_recognition.face_encodings(image, bounding_boxes)
        return (bounding_boxes[0], feature_vectors[0])

    def __draw_boxes_with_names(self, image:np.ndarray, face_locations:list=None, face_names:list=None):
        """ 
        image is numpy array picture
        Draw bounding box in the picture
        """
        # Convert PIL Image to NumPy array
        # image_np = np.array(image)
        font = cv2.FONT_HERSHEY_DUPLEX
        if face_locations == None:
            image = self.resize_image(image)
            if face_names == None:
                cv2.putText(image, 'No face', (6, 620), font, 2, (255, 255, 255), 2)
                return image 
            cv2.putText(image, face_names[0], (6, 620), font, 2, (255, 255, 255), 2)
            return image 
        if face_names == None:
            image = self.resize_image(image)
            cv2.putText(image, 'Unknown', (6, 620), font, 2, (255, 255, 255), 2)
            return image 
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            # Draw name 
            image = self.resize_image(image)
            cv2.putText(image, name, (left + 6, bottom - 6), font, 2, (255, 255, 255), 2)
            break
        # Convert the modified NumPy array back to a PIL Image
        # return Image.fromarray(image)
        return image

    def face_similar(self, image:np.ndarray, tolerance:float=0.6):
        """ 
        Check the only face in image is in database or not 

        :param image: The image contain only one object person - crop from big image in step use YOLO to detect object
        :param tolerance: The smaller make more correct when recognition 
        :return: True if the face be detect in database, False if not  
        """
        # Convert PIL Image to NumPy array (PIL Image already be converted to 'RGB' mode)
        # image_np = np.array(image) 
        image = np.ascontiguousarray(image)
        face_to_check = self.__face_rec(image)
        if face_to_check is None:
            image = self.__draw_boxes_with_names(image)
            return False, 'No face', image
        
        # for name, vector in self.face_db_regis.items():
        #     # So sánh khuôn mặt trong ảnh với tất cả khuôn mặt trong cơ sở dữ liệu
        #     match = face_recognition.compare_faces([vector], face_to_check[1], tolerance=tolerance)[0]

        #     if match:
        #         image = self.__draw_boxes_with_names(image, [face_to_check[0]], [name])
        #         return True, name, image

        matches = face_recognition.compare_faces(self.face_db_regis["encodings"], face_to_check[1], tolerance)
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = self.face_db_regis["names"][i]
                counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)
            image = self.__draw_boxes_with_names(image, [face_to_check[0]], [name])
            return True, name, image
        else:
            image = self.__draw_boxes_with_names(image, [face_to_check[0]])
            return False, 'Unknown', image
     
    @staticmethod
    def resize_image(image:np.ndarray, target_height:int=640, target_width:int=320):
        """
        :param target_height: row's number
        :param target_width: col's number
        :return numpy array of picture for height and width provided
        """
        resized_img = cv2.resize(image, (target_width, target_height))
        return resized_img
    
    @staticmethod
    def black_frame(height:int, width:int):
        """
        :param height: row's number
        :param width: col's number
        :return numpy array with zero value for height and width provided
        """
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    @staticmethod
    def take_cam(cam1:str='/home/harito/Videos/Cam1.mp4', cam2:str='/home/harito/Videos/Cam2.mp4'):
        return cv2.VideoCapture(cam1), cv2.VideoCapture(cam2)
    
    def execute(self):
        previous, current = set(), set()
        cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
        cap1, cap2 = App.take_cam()
    
        frame_number = 0 
        while cap1.isOpened() and cap2.isOpened():
            terminal_frame = App.black_frame(330, 2560)
            list_frame = [App.black_frame(640, 320)] * 8
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            boxes_cam1, track_cam1, ids1 = self.detect_track_objects(frame1, 1)
            boxes_cam2, track_cam2, ids2 = self.detect_track_objects(frame2, 2)
            if boxes_cam1 == None or boxes_cam2 == None:
                continue
            current = set(ids1 + ids2)

            track_frame = cv2.hconcat([track_cam1, track_cam2])

            # if frame_number % 24 == 0:
            for i in range(len(boxes_cam1)):
                if i >= 4: break
                # list_frame[i] = self.resize_image(self.face_similar(boxes_cam1[i][0])[2])
                list_frame[i] = self.face_similar(boxes_cam1[i][0])[2]
            for i in range((len(boxes_cam2))):
                if i >= 4: break
                # list_frame[4 + i] = self.resize_image(self.face_similar(boxes_cam2[i][0])[2])
                list_frame[4+i] = self.face_similar(boxes_cam2[i][0])[2]

            recognition_frame = cv2.hconcat(list_frame)
            output_lines = [
                f'Person(s) {previous - current} disappear from frame {frame_number}th',
                f'Person(s) {previous & current} remain in frame {frame_number}th',
                f'Person(s) {current - previous} go in from frame {frame_number}th'
            ]

            # Vị trí bắt đầu để vẽ từng dòng
            y_position = 100
            for line in output_lines:
                cv2.putText(terminal_frame, line, (15, y_position), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
                y_position += 100  # Tăng y_position để di chuyển xuống dòng tiếp theo
            
            frame = cv2.vconcat([track_frame, recognition_frame, terminal_frame])
            frame_number += 1
            previous = current
            cv2.imshow('Cam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

app = App()
app.execute()
