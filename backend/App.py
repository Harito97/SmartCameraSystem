from ultralytics import YOLO
# from yolov5 import YOLOv5
from ultralytics.engine.results import Results
import torch
from torchvision import models, transforms
import face_recognition
import pickle
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
import torchreid
# from typing import overload

class App:
    def __init__(self):
        print("Setting YOLO model for detect ...")
        # self.yolo_model = YOLO('yolov8x.pt')
        self.yolo_model = YOLO('yolov8n.pt')
        # self.yolo_model = YOLOv5('crowdhuman_yolov5m.pt')
        # print("Setting reid model: resnet50 ...") 
        # self.reid_model = models.resnet50(pretrained=True) # older way in older version of the library
        # self.reid_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        # self.reid_model.eval()
        print("Set reid with histogram") 
        print("Setting the dictionary {name: vector feature} from load_data_base() ...")
        # self.face_db_regis = App.load_data_base("database/face_dictionary.txt")
        self.face_db_regis = pickle.loads(open("database/face_dictionary_extra.pkl", "rb").read())
        print("Set face recognition model with face_recognition \nSetup done!")

    # Step 0: Load database 
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
    def __extract_features(self, image):
        """
        Para:
            image: PIL Image - picture of a single object be convert('RGB') yet
            or 
            // image: np.ndarray - 3D array representing an RGB image
        Result:
            object_images: a list of single object image detect crop from bounding box 

        How to use - Eg without OOP: 
            resnet_model = models.resnet50(pretrained=True)
            resnet_model.eval()
            img1 = Image.open("/path/to/img").convert('RGB')
            features1 = extract_features(img1, resnet_model)
        """
        # preprocess = transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])

        # input_tensor = preprocess(image)
        # input_tensor = torch.unsqueeze(input_tensor, 0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Chuyển định dạng màu BGR sang RGB

        # Bước 2: Chuyển đổi ảnh thành tensor và tiền xử lý
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # Chuyển đổi thành tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Tiền xử lý chuẩn hóa
        ])

        input_tensor = preprocess(image)
        input_tensor = input_tensor.unsqueeze(0)  # Thêm chiều batch

        with torch.no_grad():
            # output = self.reid_model(input_tensor)
            output = self.reid_model.forward(input_tensor)

        if len(output) == 0:
            return None
        return output

    def __reid_similar(self, features1, features2, threshold):
        """
        Para: 
            features1, features2: output of extract_features(image, reid_model) function - it is feature vector get after object go throw model
            threshold: double - the level consider 2 object is the same or not 
        Result:
            True if 2 object is the same, False otherwise 
        """
        return cosine_similarity(features1, features2) >= threshold

    """
    # def same_object(self, image1, image2, threshold = 0.8):
    #     ""
    #     Para:
    #         image1, image2: PIL Image - picture of a single object be convert('RGB') yet
    #     Result:
    #         True if 2 object in 2 picture is the same, False if not
    #     How to use - Eg without OOP: 
    #         img1 = Image.open("/path/to/img").convert('RGB')
    #         is_same = same_object(img1, resnet_model)
    #     ""
    #     if image1 is None or image2 is None:
    #         return False
    #     feature1 = self.__extract_features(image1)
    #     feature2 = self.__extract_features(image2)
    #     return self.__reid_similar(feature1, feature2, threshold)
    """
    def extract_features(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    def same_object(self, img1, img2, threshold = 0.8):
        # img1 = cv2.imread(img1)
        # img2 = cv2.imread(img2)
    
        # Chuyển đổi ảnh sang không gian màu HSV
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
        # Tính histogram của ảnh
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
        # Chuẩn hóa histogram
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
        # Tính độ tương đồng giữa hai histogram
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
        if similarity > threshold:
            return True
        else:
            return False

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
    