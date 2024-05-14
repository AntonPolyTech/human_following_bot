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
        
        self.subscription3 = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.subscription3

        # TF
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.transform_listener.TransformListener(self.tf_buffer, self)

        # Pub
        self.img_publisher_ = self.create_publisher(Image, '/yolo_detection', 10)
        self.rgb_depth_publisher_ = self.create_publisher(RgbDepth, '/rgb_depth_sync', 10)
        self.rgb_pc2_publisher_ = self.create_publisher(RgbPc2, '/rgb_pointcloud2', 10)
        self.goal_pose_publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.point_publisher_ = self.create_publisher(PointStamped, '/point', 10)

        # Time
        T_p_ = self.get_clock().now()
        T_p_ # to prevent unused variable warning

        # Sync
        self.image_sub_ = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub_ = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts_ = message_filters.TimeSynchronizer([self.image_sub_, self.depth_sub_], 10)
        self.ts_.registerCallback(self.cam_callback)

        # Variables
        self.odom_ = Odometry()
        self.map_ = OccupancyGrid()
        self.P_2D_ = []
        self.mask_ = []
        self.P_3D_ = []
        self.bridge_ = CvBridge()

        # Wheights for NN
        self.MODEL_ = YOLO('weights/yolov8n-seg.engine')

    def getBaseLinkAngles(self):
        try:
            transform = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
            q = Quaternion()
            q.x = transform.transform.rotation.x
            q.y = transform.transform.rotation.y
            q.z = transform.transform.rotation.z
            q.w = transform.transform.rotation.w
            angles = euler_from_quaternion(q)

            return angles
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {1} to {1}: {ex}')
            return []

    def getBaseLinkPosition(self):
        try:
            print("one")
            transform = self.tf_buffer.lookup_transform('base_link', 'map', rclpy.time.Time())
            
            t = Vector3()
            t.x = transform.transform.translation.x
            t.y = transform.transform.translation.y
            t.z = transform.transform.translation.z
            print("two")

            return t
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {1} to {1}: {ex}')
            return []

    def calcRobotXY(self):
        h = np.sqrt(np.square(self.P_3D_[2]) + np.square(self.P_3D_[0]))
        x = ((h - self.SAFE_DISTANCE_) * self.P_3D_[2]) / h
        y = ((h - self.SAFE_DISTANCE_) * -self.P_3D_[0]) / h
        return x, y

    def getCoordsFromPC2(self, _data, _x, _y):
        xyz_array = pointcloud2_to_xyz_array(_data.pc2)
        P_3D = xyz_array[_data.rgb.width * _y + _x]
        return P_3D
    
    def findMedianValue(self, _data): #, _ul, _lr):
        array = []
        
        # for y in range(_ul[1], _lr[1], 2):
        #     for x in range(_ul[0], _lr[0], 2):
        #         d = _data[]
        #         if d != 0:
        #             array.append(d)
        
        for i in range (len(_data)):
            array.append(_data[i][2])
        
        array.sort()
        n = len(array)
        if n != 0:    
            if n % 2 == 0:
                return (array[n // 2 - 1] + array[n // 2]) / 2.0
            else:
                return array[n // 2]
        else:
            return []
            
    def humanDepth(self, _data):
        array = []

        for y in range(0, _data.rgb.height, 1):
            for x in range(0, _data.rgb.width, 1):

                if self.mask_[y][x] != 0:
                    array.append(self.getCoordsFromPC2(_data, x, y))
                    # array.append([1, 1, 1])
        # print('lenARRAY =', len(array))
        # print('ARRAY =', array)
        return array

    def rgb_pc2_callback(self, _data):
        if len(self.P_2D_) == 0:
            print('P_2D is empty')
            return None
        

        xyz_array = pointcloud2_to_xyz_array(_data.pc2)
        self.P_3D_ = xyz_array[_data.rgb.width * self.P_2D_[1] + self.P_2D_[0]]

        humanDepth = self.humanDepth(_data)
        self.P_3D_[2] = self.findMedianValue(humanDepth)
        print('P_3D = ', self.P_3D_)


        if len(self.P_3D_) != 0:

            goal_pose = PoseStamped()
            
            # Goal pose header
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Goal pose local coordinates
            goal_pose_local_coords = Point()
            
            x, y = self.calcRobotXY()

            goal_pose_local_coords.x = x
            goal_pose_local_coords.y = y
            goal_pose_local_coords.z = self.odom_.pose.pose.position.z

            # Goal pose point for multiplication
            goal_pose_local_point = PointStamped()
            goal_pose_local_point.header.frame_id = 'base_link'
            goal_pose_local_point.point = goal_pose_local_coords

            # Transform coordinates from base_link to map frame
            bl2map = self.getTransform()
            goal_pose_global_point = do_transform_point(goal_pose_local_point, bl2map)
            target_real_coords = goal_pose_global_point.point

            # Goal pose position

            bl_coords = self.getBaseLinkPosition()  # base_link coords in map coords

            goal_pose.pose.position.x = target_real_coords.x
            goal_pose.pose.position.y = target_real_coords.y
            goal_pose.pose.position.z = target_real_coords.z

            # Goal pose orientation
            curr_robot_angles = self.getBaseLinkAngles()
            if len(curr_robot_angles) == 3:
                curr_robot_yaw = curr_robot_angles[2]
        
                q = quaternion_from_euler(0, 0, -curr_robot_yaw - self.calcYaw())

                goal_pose.pose.orientation.x = q.x
                goal_pose.pose.orientation.y = q.y
                goal_pose.pose.orientation.z = q.z
                goal_pose.pose.orientation.w = q.w

                self.goal_pose_publisher_.publish(goal_pose)

    def cam_callback(self, _rgb, _depth):
        rgb = self.bridge_.imgmsg_to_cv2(_rgb, 'bgr8')
        cv2.imshow('RGB', rgb)
        cv2.waitKey(20)

        tracker, self.mask_, self.P_2D_ = self.human_tracker(rgb)
        img_msg = self.bridge_.cv2_to_imgmsg(tracker)
        self.img_publisher_.publish(img_msg)

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
        
        return frame_, m, P_2D
    
    def odom_callback(self, msg : Odometry):
        self.odom_ = msg
    
    def map_callback(self, msg : OccupancyGrid):
        self.map_ = msg

    def calcYaw(self):
        x = self.P_3D_[0]
        z = self.P_3D_[2]

        yaw_angle = np.arctan(x / z)  # radians
        return yaw_angle

    def getTransform(self):
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

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

            return transform_stamped
        except TransformException as ex:
            self.get_logger().info(
                        f'Could not transform {1} to {1}: {ex}')
            pass
        return TransformStamped()
    
     

def main(args=None):

    # Camera node
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()