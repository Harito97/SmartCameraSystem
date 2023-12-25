# Import thư viện
import cv2
from kafka import KafkaProducer
import threading

# Khởi tạo hàm đọc video và gửi thông tin lên Topic
def public_video_to_kafka(producer, topic, video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()

        # Publish frame to Kafka topic
        producer.send(topic, value=data)

    cap.release()

# Khởi tạo hàm main 
def main():
    # Configure Kafka producer
    bootstrap_servers = 'localhost:9092'
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                             api_version=(0, 10, 1))    

    # Create and run Kafka producer for topic 1 in a separate thread
    topic1 = 'Cam1'
    video_path1 = 'roads.mp4'
    producer_thread1 = threading.Thread(target=public_video_to_kafka,
                                        args=(producer, topic1, video_path1))
    producer_thread1.start()

    # Create and run Kafka producer for topic 2 in a separate thread
    topic2 = 'Cam2'
    video_path2 = 'roads.mp4'
    producer_thread2 = threading.Thread(target=public_video_to_kafka,
                                        args=(producer, topic2, video_path2))
    producer_thread2.start()

    producer_thread1.join()
    producer_thread2.join()

    # bootstrap_servers = 'localhost:9092'
    # producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
    #                          api_version=(0, 10, 1)) 
    # topic2 = 'topicb'
    # video_path2 = 'roads.mp4'
    # public_video_to_kafka(producer, topic2, video_path2)

if __name__ == "__main__":
    main()