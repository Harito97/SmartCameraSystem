# Import library
import cv2
from io import BytesIO
from kafka import KafkaConsumer
import numpy as np
from App import App
from PIL import Image

def consume_video_from_kafka(consumer, topic):
    # Khởi tạo hàm “consume_video_from_kafka”:
    # Đọc dữ liệu được gửi từ “topica” và
    # yêu cầu hiển thị ra màn hình video gốc
    # Tạo cửa số nhờ OpenCV để hiện thị các frame ảnh
    cv2.namedWindow(topic, cv2.WINDOW_NORMAL)

    for message in consumer:
        frame_data = np.frombuffer(message.value, dtype=np.uint8)
        frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

        # Display video with the window name as the topic
        cv2.imshow(topic, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def detect(topic, app: App, pil_image):
    """
    :param pil_image: PIL Image (the frame picture get from topic)
    """
    object_images, results = app.detect_objects(pil_image)
    return topic, object_images, results

def main():
    topic1 = "Cam1"
    topic2 = "Cam2"

    topics = [topic1, topic2]
    consumer = KafkaConsumer(
        *topics,  # Sử dụng unpacking để truyền danh sách topics làm các đối số riêng lẻ
        bootstrap_servers=["localhost:9092"],
        api_version=(0, 10)
    )

    app = App()

    for message in consumer:
        topic = message.topic
        value = message.value

        # frame_data = np.frombuffer(value, dtype=np.uint8)
        # frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)  # np array

        # Chuyển đổi dữ liệu bytes thành đối tượng PIL Image
        pil_image = Image.open(BytesIO(value)).convert('RGB')

        output = None
        
        # Xử lý dữ liệu cho từng topic
        if topic == topic1:
            output = detect(topic, app, pil_image)

        elif topic == topic2:
            output = detect(topic, app, pil_image)
        
        


if __name__ == "__main__":
    main()
