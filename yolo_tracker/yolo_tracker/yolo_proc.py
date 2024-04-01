import rclpy
import supervision as sv
import cv2
import numpy as np

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int64MultiArray
from cv_bridge import CvBridge
from .submodules.point_cloud2 import pointcloud2_to_xyz_array


# wheights for NN
MODEL = YOLO('weights/yolov8s.pt')

_P_2D = None

bridge = CvBridge()

K = np.array([[528.434, 0, 320.5],
              [0, 528.434, 240.5],
              [0, 0, 1]])

class Camera_Subscriber(Node):

    def __init__(self):

        super().__init__('camera_subscriber')

        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10)
        self.subscription

        self.subscription2 = self.create_subscription(
            PointCloud2,
            '/my_depth_sensor_pointcloud2',
            self.pc2_callback,
            10)
        self.subscription2

        self.img_publisher = self.create_publisher(Image, "/human_tracker", 10)
        self.coordinates_2D_publisher = self.create_publisher(Int64MultiArray, "/human_2d_coordinates", 10)

    def camera_callback(self, data):
        
        global _P_2D
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        tracker, flag, _P_2D = human_tracker(frame)

        if not flag:
            return self.img_publisher.publish(data)
        img_msg = bridge.cv2_to_imgmsg(tracker)
        self.img_publisher.publish(img_msg)

    def pc2_callback(self, data):
        
        global _P_2D
        
        xyz_array = pointcloud2_to_xyz_array(data)
        P_3D = xyz_array[640 * _P_2D[1] + _P_2D[0]]
        print('P_3D = ', P_3D)


        


def human_tracker(frame):
    
    results = MODEL(frame, classes=0)
    frame_ = results[0].plot()
        
    boxes = results[0].boxes.xyxy.tolist()

    if not boxes:
        print("ERROR: No human detected")
        return None, False

    boxes = boxes[0]
    
    # person coordinates
    X_p = int(boxes[2] + boxes[0]) // 2
    Y_p = int(boxes[3] + boxes[1]) // 2
    
    # 2D coordinates vector
    P_2d = np.array([X_p, Y_p, 1])

    print('Human coordinates: ', P_2d)

    # drawing human body center 
    cv2.circle(frame_, (X_p, Y_p), 5, (255, 0, 0))
    return frame_, True, P_2d



# def coordinates_2D_to_3D(data):
#     return camera_Z * K.transpose() * data


# class CoordinatesPublisher(Node):

#     def __init__(self):
#         super().__init__('coordinates_publisher')
        
#         self.publisher_ = self.create_publisher(
#             np.ndarray,
#             'human_coordinates',
#             10)
        
#         self.coordinates_2d_sub = self.create_subscription(np.ndarray, "/human_2d_coordinates", 10)
        
#     def timer_callback(data):
#         coordinates_2D_to_3D(data)

# def coordinates_2D_to_3D(P_2D):
#     P_3D = K.transpose() * P_2D


def main(args=None):

    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    # human_coordinates = CoordinatesPublisher()
    rclpy.spin(camera_subscriber)
    # rclpy.spin(human_coordinates)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
