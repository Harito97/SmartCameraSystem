from App import App
import cv2
import numpy as np

class Main:
    def __init__(self, cam1:str='/home/harito/Videos/Cam1.mp4', cam2:str='/home/harito/Videos/Cam2.mp4'):
        self.app = App()
        self.cap1 = cv2.VideoCapture(cam1)
        self.cap2 = cv2.VideoCapture(cam2)
        self.reid_dict = Main.load_reid_dict()

    @staticmethod
    def load_reid_dict():
        reid_dict = {}
        loaded_data = np.load('database/histograms.npz')
        for key in loaded_data.files:
            reid_dict[key] = loaded_data[key]
        return reid_dict
    
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
    
    def detect(self, frame1, frame2):
        result1_detect = self.app.detect_objects(frame1)
        result2_detect = self.app.detect_objects(frame2)
        frame1_detect = cv2.cvtColor(np.array(self.app.draw_picture_detect(result1_detect[1])), cv2.COLOR_RGB2BGR)
        frame2_detect = cv2.cvtColor(np.array(self.app.draw_picture_detect(result2_detect[1])), cv2.COLOR_RGB2BGR)
        return result1_detect, result2_detect, cv2.hconcat([frame1_detect, frame2_detect])

    def main(self):
        cv2.namedWindow('Cam', cv2.WINDOW_NORMAL)
        # #######################################
        terminal_frame = Main.black_frame(1440, 2560)
        list_frame = [Main.black_frame(1440, 640)] * 8
        i = 0
        while self.cap1.isOpened() and self.cap2.isOpened():
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                break
            
            real_frame = cv2.hconcat([frame1, frame2])

            if i % 24 == 0:
                # Detect
                result1_detect, result2_detect, detect_frame = self.detect(frame1, frame2) 
                
             
            cv2.imshow('Cam', cv2.hconcat([cv2.vconcat([real_frame, detect_frame]), terminal_frame]))
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    Main().main()
