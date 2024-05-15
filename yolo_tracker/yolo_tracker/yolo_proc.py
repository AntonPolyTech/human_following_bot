import rclpy
import supervision as sv
import cv2
import numpy as np
import message_filters
import rclpy.time
import tf2_ros
import math

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from tracker_msgs.msg import RgbPc2, RgbDepth
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PointStamped, Vector3
from cv_bridge import CvBridge
from .submodules.point_cloud2 import pointcloud2_to_xyz_array
from nav_msgs.msg import Odometry, OccupancyGrid
# from tf_transformations import quaternion_from_euler, euler_from_quaternion
from tf2_geometry_msgs import *
from tf2_ros import Time
from tf2_ros import TransformException


def euler_from_quaternion(quaternion : Quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion (w in last place)
    quat = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    q.w = cy * cp * cr + sy * sp * sr

    return q

def equation_of_line(x1, y1, x2, y2):
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k, b

class Camera_Subscriber(Node):

    def __init__(self):

        super().__init__('camera_subscriber')

        self.SAFE_DISTANCE_ = 2.0 #meters

        self.subscription1 = self.create_subscription(
            RgbPc2,
            '/rgb_pointcloud2',
            self.rgb_pc2_callback,
            10)
        self.subscription1

        self.subscription2 = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription2
        
        # TF
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.transform_listener.TransformListener(self.tf_buffer, self)
        self.tf_ = None

        # Pub
        self.img_publisher_ = self.create_publisher(Image, '/yolo_detection', 10)
        self.rgb_depth_publisher_ = self.create_publisher(RgbDepth, '/rgb_depth_sync', 10)
        self.rgb_pc2_publisher_ = self.create_publisher(RgbPc2, '/rgb_pointcloud2', 10)
        self.goal_pose_publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.point_publisher_ = self.create_publisher(PointStamped, '/point', 10)

        # Time
        T_p_ = self.get_clock().now()
        T_p_ # to prevent unused variable warning
        timer_period = 10 / 1000  # seconds
        self.timer = self.create_timer(timer_period, self.getTransform)

        # Sync
        self.image_sub_ = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub_ = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts_ = message_filters.TimeSynchronizer([self.image_sub_, self.depth_sub_], 10)
        self.ts_.registerCallback(self.cam_callback)

        # Variables
        self.odom_ = Odometry()
        self.mask_ = []
        self.P_3D_ = []
        self.bridge_ = CvBridge()

        # Wheights for NN
        self.MODEL_ = YOLO('weights/yolov8n-seg.engine')

    def getBaseLinkAngles(self):
        transform = self.tf_
        if transform == None:
            return None

        q = Quaternion()
        q.x = transform.transform.rotation.x
        q.y = transform.transform.rotation.y
        q.z = transform.transform.rotation.z
        q.w = transform.transform.rotation.w
        angles = euler_from_quaternion(q)

        return angles

    def getBaseLinkPosition(self):
        transform = self.tf_
        if transform == None:
            return None
        
        t = Vector3()
        t.x = transform.transform.translation.x
        t.y = transform.transform.translation.y
        t.z = transform.transform.translation.z

        return t

    def calcRobotXY(self):
        h = np.sqrt(np.square(self.P_3D_[2]) + np.square(self.P_3D_[0]))
        x = ((h - self.SAFE_DISTANCE_) * self.P_3D_[2]) / h
        y = ((h - self.SAFE_DISTANCE_) * -self.P_3D_[0]) / h
        return x, y

    # def getCoordsFromPC2(self, _data, _x, _y):
    #     xyz_array = pointcloud2_to_xyz_array(_data.pc2)
    #     P_3D = xyz_array[_data.rgb.width * _y + _x]
    #     return P_3D
    
    def findMedianValue(self, _data : list): #, _ul, _lr):
        _data.sort()
        n = len(_data)

        if n != 0:
            if n % 2 == 0:
                return (_data[n // 2 - 1] + _data[n // 2]) / 2.0
            else:
                return _data[n // 2]
        else:
            return np.array([])
            
    def humanDepth(self, _data):
        flattened_array = self.mask_.flatten()
        idences = np.where(flattened_array != 0)
        result = _data[idences][:, 2]
        return result
    
    # ------------------------------------------- R G B P C 2   C A L L B A C K ------------------------------------------- 

    def rgb_pc2_callback(self, _data):
        rgb = self.bridge_.imgmsg_to_cv2(_data.rgb, 'bgr8')
        h, w = rgb.shape[:2]

        tracker, self.mask_, P_2D = self.human_tracker(rgb)
        img_msg = self.bridge_.cv2_to_imgmsg(tracker)
        self.img_publisher_.publish(img_msg)

        if self.tf_ == None:
            return

        if len(P_2D) == 0:
            print('P_2D is empty')
            return None
        
        xyz_array = pointcloud2_to_xyz_array(_data.pc2)
        self.P_3D_ = xyz_array[_data.rgb.width * P_2D[1] + P_2D[0]]

        humanDepth = self.humanDepth(xyz_array)

        if len(self.P_3D_) != 0:

            self.P_3D_[2] = self.findMedianValue(humanDepth)

            if self.P_3D_[2] <= self.SAFE_DISTANCE_:
                return None

            print('P_3D = ', self.P_3D_)

            goal_pose = PoseStamped()
            
            # Goal pose header
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Goal pose local coordinates
            goal_pose_local_coords = Point()
            
            x, y = self.calcRobotXY()

            # # # # # # # # # # # # # # # # # # # # #
            # Human point TEST 
            point = PointStamped()

            point.point.x = self.P_3D_[2]
            point.point.y = -self.P_3D_[0]
            point.point.z = self.odom_.pose.pose.position.z

            # global_point = do_transform_point(point, bl2map)

            point.header.frame_id = 'base_link'
            point.header.stamp = self.get_clock().now().to_msg()
            
            self.point_publisher_.publish(point)
            # # # # # # # # # # # # # # # # # # # # #

            goal_pose_local_coords.x = x
            goal_pose_local_coords.y = y
            goal_pose_local_coords.z = self.odom_.pose.pose.position.z

            # Goal pose point for multiplication
            goal_pose_local_point = PointStamped()
            goal_pose_local_point.header.frame_id = 'base_link'
            goal_pose_local_point.point = goal_pose_local_coords

            # Transform coordinates from base_link to map frame
            print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTf', self.tf_)
            print('2222222222222222222222222222222222_D', P_2D)
            goal_pose_global_point = do_transform_point(goal_pose_local_point, self.tf_)
            target_real_coords = goal_pose_global_point.point

            # Goal pose position
            goal_pose.pose.position.x = target_real_coords.x
            goal_pose.pose.position.y = target_real_coords.y
            goal_pose.pose.position.z = target_real_coords.z

            print('PPPPPPPPPPPPPpose', target_real_coords)

            # Goal pose orientation
            curr_robot_angles = self.getBaseLinkAngles()
            if len(curr_robot_angles) == 3:
                curr_robot_yaw = curr_robot_angles[2]

                q = quaternion_from_euler(0, 0, curr_robot_yaw - self.calcYaw())

                goal_pose.pose.orientation.x = q.x
                goal_pose.pose.orientation.y = q.y
                goal_pose.pose.orientation.z = q.z
                goal_pose.pose.orientation.w = q.w

                self.goal_pose_publisher_.publish(goal_pose)

    # ------------------------------------------- C A M E R A   C A L L B A C K ------------------------------------------- 

    def cam_callback(self, _rgb, _depth):
        rgb_depth = RgbDepth()
        rgb_depth.rgb = _rgb
        rgb_depth.depth = _depth
        self.rgb_depth_publisher_.publish(rgb_depth)
 
    def human_tracker(self, frame):
        results = self.MODEL_.predict(source=frame, classes=0, save=False)
        frame_ = results[0].plot()

        if results[0].masks == None:
            return frame, [], []
        
        m = results[0].masks.data[0].cpu().numpy() * np.uint(255)        
        boxes = results[0].boxes.xyxy.tolist()

        if not boxes and not results[0].masks.data[0]:
            print("ERROR: No human detected")
            return frame, [], []
        
        boxes = boxes[0]
        
        # person coordinates
        X_p = int(boxes[2] + boxes[0]) // 2
        Y_p = int(boxes[3] + boxes[1]) // 2
        
        # 2D coordinates vector
        P_2D = np.array([X_p, Y_p, 1])

        print('Human pixel coordinates: ', P_2D)

        # drawing human body center 
        cv2.circle(frame_, (X_p, Y_p), 5, (255, 0, 0), 10)
        
        return frame_, m, P_2D
    
    def odom_callback(self, msg : Odometry):
        self.odom_ = msg

    def calcYaw(self):
        x = self.P_3D_[0]
        z = self.P_3D_[2]

        yaw_angle = np.arctan(x / z)  # radians
        return yaw_angle

    def getTransform(self):
        print('GGGGGGGGGGGGGGGGGGGGGGGGET')
        try:
            print('EEEEEEEEEEEEEEEEmpty')
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            print('LLLLLLLLLLLokup')
            r = Quaternion()
            r.x = transform.transform.rotation.x
            r.y = transform.transform.rotation.y
            r.z = transform.transform.rotation.z
            r.w = transform.transform.rotation.w

            t = Vector3()
            t.x = transform.transform.translation.x
            t.y = transform.transform.translation.y
            t.z = transform.transform.translation.z

            transform_stamped = TransformStamped()
            transform_stamped.transform.rotation = r
            transform_stamped.transform.translation = t

            self.tf_ =  transform_stamped
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {1} to {1}: {ex}')
    
     

def main(args=None):

    # Camera node
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()