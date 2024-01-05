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
        self.yolo_model = YOLO('yolov8n.pt')
        print("Setting YOLO model for detect ...")
        self.track_history = defaultdict(lambda: [])
        print('Setting track history ...')
        self.face_db_regis = pickle.loads(open("data/face_dictionary_extra.pkl", "rb").read())
        print("Set face recognition model with face_recognition \nSetup done!")

    # Detect and Track objects 
    def detect_track_objects(self, image:np.ndarray):
        results = self.yolo_model.track(source=image, persist=True, classes=0)
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        bounding_boxes = []
        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = self.track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 50: 
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 230, 0), thickness=10)

            # Get bounding box and it id 
            xmin = int(x - w / 2)
            ymin = int(y - h / 2)
            xmax = int(x + w / 2)
            ymax = int(y + h / 2)
            bounding_boxes.append((image[ymin:ymax, xmin:xmax, :], track_id))

        return bounding_boxes, annotated_frame

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
    
    