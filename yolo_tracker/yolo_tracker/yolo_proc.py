import rclpy
import supervision as sv
import cv2
import numpy as np

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64MultiArray
from cv_bridge import CvBridge
from .submodules.point_cloud2 import pointcloud2_to_xyz_array


# wheights for NN
MODEL = YOLO('weights/yolov8s.pt')

_P_2D = None
_P_3D = None


T_p = 0

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

        self.subscription3 = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.EKF_callback,
            10)
        self.subscription3

        self.img_publisher = self.create_publisher(Image, "/human_tracker", 10)
        # self.coordinates_2D_publisher = self.create_publisher(Int64MultiArray, "/human_2d_coordinates", 10)
        T_p = self.get_clock().now()
        T_p # to prevent unused variable warning
        self.is_first = True
        self.null_box = False

    def camera_callback(self, data):
        if not self.null_box:
            return
        global _P_2D
        frame = bridge.imgmsg_to_cv2(data, "bgr8")
        tracker, flag, _P_2D = human_tracker(frame)
        if _P_2D == None:
            return

        if not flag:
            return self.img_publisher.publish(data)
        img_msg = bridge.cv2_to_imgmsg(tracker)
        self.img_publisher.publish(img_msg)

    def pc2_callback(self, data):
        if not self.null_box:
            return
        
        global _P_2D, _P_3D
        if _P_2D == None:
            return None
        
        xyz_array = pointcloud2_to_xyz_array(data)
        _P_3D = xyz_array[640 * _P_2D[1] + _P_2D[0]]
        print('P_3D = ', _P_3D)


    def EKF_callback(self, msg: Twist):
        if not self.null_box:
            return
        
        global _P_3D
        if _P_3D == None:
            return
        
        if self.is_first:
            self.is_first = False
            x_h_prev = _P_3D[0]
            y_h_prev = _P_3D[2]
            return

        # Ron=bot velocities
        w_robot = msg.angular.z
        V_robot = msg.linear.x
        
        # Time counting
        T = self.get_clock().now()
        delta_T = T - T_p
        T_p = T

        # W_k matrix
        teta = w_robot * delta_T
        cos_teta = np.cos(teta)
        sin_teta = np.sin(teta)

        l_x = V_robot * sin_teta
        l_y = V_robot * cos_teta

        W_k = np.array([[cos_teta, sin_teta, 0, 0],
                        [-sin_teta, cos_teta, 0, 0],
                        [0, 0, 1, 0],
                        [l_x, l_y, 0, 1]])
        
        # Human previous coordinates h(k)
        h_k = np.array([_P_3D[0], _P_3D[2], 1, 1])

        # Human linear velocities
        V_hx = (h_k[0] - x_h_prev) / delta_T 
        V_hy = (h_k[1] - y_h_prev) / delta_T 
        
        x_h_prev = h_k[0]
        y_h_prev = h_k[1]

        # Human predicted coordinates h(k+1)
        x_h_predicted = h_k[0] + V_hx * delta_T
        y_h_predicted = h_k[1] + V_hy * delta_T

        h_k_predicted = np.array([x_h_predicted, y_h_predicted, 1, 1])

        # Predicted local coordinates
        h_predicted = h_k_predicted * np.linalg.inv(W_k)
        print('h_predicted: ', h_predicted)

        return h_predicted

    
def human_tracker(self, frame):
    
    results = MODEL(frame, classes=0)
    frame_ = results[0].plot()
        
    boxes = results[0].boxes.xyxy.tolist()

    if not boxes:
        self.null_box = True
        print("ERROR: No human detected")
        return None, False, None
    else:
        self.null_box = False

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


def main(args=None):

    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
