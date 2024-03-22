import rclpy
import supervision as sv
import cv2

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# wheights for NN
MODEL = YOLO('weights/yolov8s.pt')

bridge = CvBridge()

class Camera_Subscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)
        self.subscription

        self.img_publisher = self.create_publisher(Image, "/human_tracker", 1)

    def camera_callback(self, data):
        global frame
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        tracker = human_tracker(frame)
        img_msg = bridge.cv2_to_imgmsg(tracker)
        self.img_publisher.publish(img_msg)

def human_tracker(frame):
    results = MODEL(frame, classes=0)
    frame_ = results[0].plot()
    
    # boxes = results[0].boxes.xyxy.tolist()
    # boxes = boxes[0]
    
    # # person coordinates
    # X_p = int(boxes[2] + boxes[0]) // 2
    # Y_p = int(boxes[3] + boxes[1]) // 2
    
    # # drawing human body center 
    # cv2.circle(frame_, (X_p, Y_p), 5, (255, 0, 0))
    return frame_

def main(args=None):
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    # camera_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
