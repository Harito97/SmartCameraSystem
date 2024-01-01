from typing import overload
from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
from torchvision import transforms
import face_recognition
import pickle
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

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
        print("Setting YOLO model for detect ...")
        self.yolo_model = YOLO('yolov8n.pt')
        print("Set reid with histogram") 
        self.reid_dict = App.load_reid_dict()
        print("Setting the dictionary {name: vector feature} from load_data_base() ...")
        self.face_db_regis = pickle.loads(open("data/face_dictionary_extra.pkl", "rb").read())
        print("Set face recognition model with face_recognition \nSetup done!")

    # Step 0: Load database
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
    
    @staticmethod
    def load_data_base(new_db: str, face_db_regis: dict = None):
        """ 
        Load new database about face user 
        Attribute face_db_regis have type dict{name (str): vector feature of the face with that name (numpy array from face_recognition.face_encodings)}
        If face_db_regis is None then return a new dict{name (str): vector feature of the face with that name (numpy array from face_recognition.face_encodings)}
        """
        if face_db_regis is not None:
            # Handle the case with two arguments
            with open(new_db, "r") as file:
                lines = file.readlines()
                for line in lines:
                    person_name, face_encoding_str = line.split(":")
                    face_encoding = np.array(eval(face_encoding_str.strip()))
                    face_db_regis[person_name.strip()] = face_encoding
        else:
            # Handle the case with one argument
            face_db_regis = {}
            with open(new_db, "r") as file:
                lines = file.readlines()
                for line in lines:
                    person_name, face_encoding_str = line.split(":")
                    face_encoding = np.array(eval(face_encoding_str))  # Sử dụng eval để chuyển đổi chuỗi thành list
                    face_db_regis[person_name.strip()] = face_encoding
            return face_db_regis

    # Step 1: Detect object with YOLO 
    def detect_objects(self, image):
        """
        Para:
            image: PIL Image - be convert('RGB') yet or np.ndarray
        Result:
            object_images: a list of single object (only person) image detect crop from bounding box 
            results: the engine.results.Results after going through the YOLO model 
        How to use - Eg without OOP:
            from ultralytics import YOLO 
            yolo_model = YOLO('yolov8x.pt') 
            img1 = Image.open("/path/to/img").convert('RGB')
            object_images_1, results = detect_objects(img1, yolo_model)
        """
        # Use YOLO to determine the regions containing objects
        results = self.yolo_model(image)
        # Get information about objects and bounding boxes
        boxes = results[0].boxes
        # Cut and save the regions containing objects
        object_images = []
        for i in range(len(boxes)):
            if boxes[i].cls.item() == 0.0:
                data = boxes[i].xywh[0]
                x_center, y_center, width, height = data[0].item(), data[1].item(), data[2].item(), data[3].item()
                xmin = int(x_center - width / 2)
                ymin = int(y_center - height / 2)
                xmax = int(x_center + width / 2)
                ymax = int(y_center + height / 2)
                # Kiểm tra loại dữ liệu của object_image
                if isinstance(image, np.ndarray):
                    # Nếu object_image là numpy.ndarray
                    object_image = image[ymin:ymax, xmin:xmax, :]
                elif isinstance(image, Image.Image):
                    # Nếu object_image là PIL.Image.Image
                    object_image = image.crop((xmin, ymin, xmax, ymax))
                else:
                    # Nếu không phải kiểu dữ liệu mong đợi, raise TypeError
                    raise TypeError("Unsupported object_image type")
                object_images.append(object_image)

        return object_images, results
    
    def draw_picture_detect(self, results: Results):
        # for r in results:
        r = results[0]
        im_array = r.plot()  # plot a BGR numpy array of predictions
        
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        # im.save('results.jpg')  # save image
        return im

    # Step 2: ReID model take the same object in each camera as 1
    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    def same_object(self, img:np.ndarray, reid_picture:dict, threshold=0.8):
        img_feature = self.extract_features(img)
        for key in self.reid_dict.keys():
            for data in self.reid_dict[key].data:
                if cv2.compareHist(data, img_feature, cv2.HISTCMP_CORREL) > threshold:
                    self.reid_dict[key].add(img_feature)
                    reid_picture[key].add(App.resize_image(img))
                    return True, reid_picture
        return False, reid_picture

    # Step 3: Face recognition from each object crop and id by reid model 
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
            if face_names == None:
                cv2.putText(image, 'No face', (6, 1400), font, 5, (255, 255, 255), 4)
                return image 
            cv2.putText(image, face_names[0], (6, 1400), font, 5, (255, 255, 255), 4)
            return image 
        if face_names == None:
            cv2.putText(image, 'Unknown', (6, 1400), font, 5, (255, 255, 255), 4)
            return image 
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw bounding box
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 5)

            # Draw name 
            cv2.putText(image, name, (left + 6, bottom - 6), font, 5, (255, 255, 255), 5)
        # Convert the modified NumPy array back to a PIL Image
        # return Image.fromarray(image)
        return image

    def face_similar(self, image:np.ndarray, tolerance:float=0.6):
        """ 
        Check the only face in image is in database or not 

        :param image: The image (PIL Image) contain only one object person - crop from big image in step use YOLO to detect object
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
    def resize_image(image, target_height=1440, target_width=640):
        resized_img = cv2.resize(image, (target_width, target_height))
        return resized_img
    
    @staticmethod
    def black_frame(height, width):
        """
        :param height: row's number
        :param width: col's number
        :return numpy array with zero value for height and width provided
        """
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    @staticmethod
    def take_cam(cam1:str='/home/harito/Videos/Cam1.mp4', cam2:str='/home/harito/Videos/Cam2.mp4'):
        return cv2.VideoCapture(cam1), cv2.VideoCapture(cam2)

    def detect(self, frame1, frame2):
        result1_detect = self.detect_objects(frame1)
        result2_detect = self.detect_objects(frame2)
        frame1_detect = cv2.cvtColor(np.array(self.draw_picture_detect(result1_detect[1])), cv2.COLOR_RGB2BGR)
        frame2_detect = cv2.cvtColor(np.array(self.draw_picture_detect(result2_detect[1])), cv2.COLOR_RGB2BGR)
        return result1_detect, result2_detect, cv2.hconcat([frame1_detect, frame2_detect])
    
    @staticmethod
    def update_list_frame(list_picture, dict_picture):
        person_names = ['Hai', 'Duc', 'DucAnh', 'Thang']

        for idx, name in enumerate(person_names):
            if len(dict_picture[name].data) >= 2:
                list_picture[idx * 2] = dict_picture[name].data[0]
                list_picture[idx * 2 + 1] = dict_picture[name].data[1]
            elif len(dict_picture[name].data) == 1:
                list_picture[idx * 2] = dict_picture[name].data[0]
                # Clear the second frame if there's only one picture available
        return list_picture

    
    def execute(self):
        cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
        # #######################################
        cap1, cap2 = App.take_cam()
        terminal_frame = App.black_frame(1440, 2560)
        list_frame = [App.black_frame(1440, 640)] * 8
        reid_picture = {'Hai': List(), 'Duc': List(), 'DucAnh': List(), 'Thang': List()}
        i = 0
        while cap1.isOpened() and cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break
            
            real_frame = cv2.hconcat([frame1, frame2])

            if i % 24 == 0:
                # Detect
                result1_detect, result2_detect, detect_frame = self.detect(frame1, frame2) 
                real_detect_frame = cv2.vconcat([real_frame, detect_frame])
                if i % 10 == 0:
                    for per1 in result1_detect[0]:
                        if self.same_object(per1, reid_picture):
                            break
                    for per2 in result2_detect[0]:
                        if self.same_object(per2, reid_picture):
                            break
                    list_frame = App.update_list_frame(list_frame, reid_picture)
                    reid_frame = cv2.hconcat(list_frame)

            # cv2.imshow('Cam', cv2.hconcat([cv2.vconcat([real_frame, detect_frame]), terminal_frame]))
            frame = cv2.vconcat(cv2.hconcat([real_detect_frame, terminal_frame]), reid_frame)
            cv2.imshow('Cam', frame)
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

# app = App()
# app.execute()
