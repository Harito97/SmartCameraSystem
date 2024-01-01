# Import library
import cv2
from io import BytesIO
from kafka import KafkaConsumer
import numpy as np
from App import App
from PIL import Image

# def detect(app: App, pil_image):
#     """
#     :param pil_image: PIL Image (the frame picture get from topic)
#     """
#     object_images, results = app.detect_objects(pil_image)
#     return object_images, results

def main():
    topic1 = "Cam1"
    topic2 = "Cam2"

    topics = [topic1, topic2]
    consumer = KafkaConsumer(
        *topics,  # Sử dụng unpacking để truyền danh sách topics làm các đối số riêng lẻ
        bootstrap_servers=["localhost:9092"],
        api_version=(0, 10)
    )
    print('Started consumer get data from 2 topic')
    print('Starting app ...')
    app = App()

    cv2.namedWindow('Cam1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cam2', cv2.WINDOW_NORMAL)
    output1, output2 = None, None

    for message in consumer:
        topic = message.topic
        value = message.value

        # Chuyển đổi dữ liệu bytes thành đối tượng PIL Image
        pil_image = Image.open(BytesIO(value)).convert('RGB')

        # Xử lý dữ liệu cho từng topic
        if topic == topic1:
            print('Get a message from topic1')
            output1 = app.detect_objects(pil_image)
        elif topic == topic2:
            print('Get a message from topic2')
            output2 = app.detect_objects(pil_image)

        # same_object = []
        if output1 is not None and output2 is not None:
            # for object_image1 in output1[0]:
            #     for object_image2 in output2[0]:
            #         if app.same_object(object_image1, object_image2):
            #             print('Have a pair same object')
            #             same_object.append((object_image1, object_image2))
            # Show Cam1 and Cam2
            frame1 = cv2.cvtColor(np.array(app.draw_picture_detect(output1[1])), cv2.COLOR_RGB2BGR)
            cv2.imshow('Cam1', frame1)
            print('Showing Cam1')
            frame2 = cv2.cvtColor(np.array(app.draw_picture_detect(output2[1])), cv2.COLOR_RGB2BGR)
            cv2.imshow('Cam2', frame2)
            print('Showing Cam2')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
