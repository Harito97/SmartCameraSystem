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
    print('Created producer')
    
    topic1 = 'Cam1'
    video_path1 = '/home/harito/Videos/Cam1.mkv'
    producer_thread1 = threading.Thread(target=public_video_to_kafka,
                                        args=(producer, topic1, video_path1))
    print('Started topic1: Cam1')
    topic2 = 'Cam2'
    video_path2 = '/home/harito/Videos/Cam2.webm'
    producer_thread2 = threading.Thread(target=public_video_to_kafka,
                                        args=(producer, topic2, video_path2))
    print('Started topic2: Cam2')

    producer_thread1.start()
    print('Sending data to topic1: Cam1')
    producer_thread2.start()
    print('Sending data to topic2: Cam2')
    
    producer_thread1.join()
    producer_thread2.join()

    print('All done!')
    # bootstrap_servers = 'localhost:9092'
    # producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
    #                          api_version=(0, 10, 1)) 
    # topic2 = 'topicb'
    # video_path2 = 'roads.mp4'
    # public_video_to_kafka(producer, topic2, video_path2)

if __name__ == "__main__":
    main()