import rclpy
import supervision as sv
import cv2
import numpy as np
import message_filters

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from tracker_msgs.msg import RgbPc2, RgbDepth
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64MultiArray
from cv_bridge import CvBridge
from .submodules.point_cloud2 import pointcloud2_to_xyz_array


# wheights for NN
MODEL = YOLO('weights/yolov8s.pt')

human_detected = False

_P_2D = []
_P_3D = []


T_p = 0

bridge = CvBridge()

K = np.array([[528.434, 0, 320.5],
              [0, 528.434, 240.5],
              [0, 0, 1]])

class Camera_Subscriber(Node):

    def __init__(self):

        super().__init__('camera_subscriber')

        self.subscription1 = self.create_subscription(
            RgbPc2,
            '/rgb_pointcloud2',
            self.rgb_pc2_callback,
            10)
        self.subscription1

        # self.subscription3 = self.create_subscription(
        #     Twist,
        #     '/cmd_vel',
        #     self.EKF_callback,
        #     10)
        # self.subscription3

        self.img_publisher = self.create_publisher(Image, '/yolo_detection', 10)
        self.rgb_depth_publisher = self.create_publisher(RgbDepth, '/rgb_depth_sync', 10)

        T_p = self.get_clock().now()
        T_p # to prevent unused variable warning

        # Sync
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.cam_callback)


    def rgb_pc2_callback(self, data):
              
        global _P_2D, _P_3D
        if len(_P_2D) == 0:
            print('_P_2D is empty')
            return None
        
        xyz_array = pointcloud2_to_xyz_array(data.pc2)
        _P_3D = xyz_array[640 * _P_2D[1] + _P_2D[0]]
        print('P_3D = ', _P_3D)


    def cam_callback(self, _rgb, _depth):

        global _P_2D

        rgb = bridge.imgmsg_to_cv2(_rgb, 'bgr8')
        cv2.imshow('RGB', rgb)
        cv2.waitKey(20)

        tracker, _P_2D = self.human_tracker(rgb)
        img_msg = bridge.cv2_to_imgmsg(tracker)
        self.img_publisher.publish(img_msg)

        rgb_depth = RgbDepth()
        rgb_depth.rgb = _rgb
        rgb_depth.depth = _depth
        self.rgb_depth_publisher.publish(rgb_depth)

 
    def human_tracker(self, frame):
        
        results = MODEL(frame, classes=0)
        frame_ = results[0].plot()
            
        boxes = results[0].boxes.xyxy.tolist()

        if not boxes:
            print("ERROR: No human detected")
            return frame, []

        boxes = boxes[0]
        
        # person coordinates
        X_p = int(boxes[2] + boxes[0]) // 2
        Y_p = int(boxes[3] + boxes[1]) // 2
        
        # 2D coordinates vector
        P_2d = np.array([X_p, Y_p, 1])

        print('Human pixel coordinates: ', P_2d)

        # drawing human body center 
        cv2.circle(frame_, (X_p, Y_p), 5, (255, 0, 0))
        return frame_, P_2d
    
    def find_p3D(self, data):
      
        global _P_2D, _P_3D
        if _P_2D == None:
            return None
        
        xyz_array = pointcloud2_to_xyz_array(data)
        _P_3D = xyz_array[640 * _P_2D[1] + _P_2D[0]]
        print('P_3D = ', _P_3D)


def main(args=None):

    # Camera node
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
