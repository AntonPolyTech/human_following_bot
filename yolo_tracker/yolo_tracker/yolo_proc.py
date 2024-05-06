import rclpy
import supervision as sv
import cv2
import numpy as np
import message_filters

from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from tracker_msgs.msg import RgbPc2, RgbDepth
from geometry_msgs.msg import Twist, PoseStamped, Quaternion
from cv_bridge import CvBridge
from .submodules.point_cloud2 import pointcloud2_to_xyz_array
from nav_msgs.msg import OccupancyGrid, Odometry

# wheights for NN
MODEL = YOLO('weights/yolov8n.pt')

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

        self.SAFE_DISTANCE_ = 2 #meters

        self.subscription1 = self.create_subscription(
            RgbPc2,
            '/rgb_pointcloud2',
            self.rgb_pc2_callback,
            10)
        self.subscription1

        self.subscription2 = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription2
        
        self.subscription3 = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription3
        
        self.subscription4 = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)
        self.subscription4
        
        self.subscription5 = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.occupancy_grid_callback,
            10)
        self.subscription5
        
        # Pub
        self.img_publisher_ = self.create_publisher(Image, '/yolo_detection', 10)
        self.rgb_depth_publisher_ = self.create_publisher(RgbDepth, '/rgb_depth_sync', 10)
        self.goal_pose_publisher_ = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.local_map_publisher_ = self.create_publisher(OccupancyGrid, '/map', 10)
        self.map_img_publisher_ = self.create_publisher(Image, '/img_map', 10)


        T_p_ = self.get_clock().now()
        T_p_ # to prevent unused variable warning

        # Sync
        self.image_sub_ = message_filters.Subscriber(self, Image, '/camera/image_raw')
        self.depth_sub_ = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts_ = message_filters.TimeSynchronizer([self.image_sub_, self.depth_sub_], 10)
        self.ts_.registerCallback(self.cam_callback)

        # Variables
        self.odom_ = Odometry()

    def rgb_pc2_callback(self, data):
              
        global _P_2D, _P_3D
        if len(_P_2D) == 0:
            print('_P_2D is empty')
            return None
        
        xyz_array = pointcloud2_to_xyz_array(data.pc2)
        _P_3D = xyz_array[data.rgb.width * _P_2D[1] + _P_2D[0]]
        print('P_3D = ', _P_3D)

        # pose = PoseStamped() : TODO: Change coordinates according to Gazebo axes
        # pose.pose.position.x = _P_3D[0]
        # pose.pose.position.y = _P_3D[1]
        # pose.pose.position.z = self.SAFE_DISTANCE_

        # q = Quaternion()
        # q.setRPY(0, 0, 0)
        # pose.pose.orientation.x = q.x()
        # pose.pose.orientation.x = q.y()
        # pose.pose.orientation.x = q.z()
        # pose.pose.orientation.x = q.w()


    def cam_callback(self, _rgb, _depth):

        global _P_2D

        rgb = bridge.imgmsg_to_cv2(_rgb, 'bgr8')
        cv2.imshow('RGB', rgb)
        cv2.waitKey(20)

        tracker, _P_2D = self.human_tracker(rgb)
        img_msg = bridge.cv2_to_imgmsg(tracker)
        self.img_publisher_.publish(img_msg)

        rgb_depth = RgbDepth()
        rgb_depth.rgb = _rgb
        rgb_depth.depth = _depth
        self.rgb_depth_publisher_.publish(rgb_depth)
 
    def human_tracker(self, frame):
        
        results = MODEL(frame, classes=0)
        frame_ = results[0].plot()
            
        boxes = results[0].boxes.xyxy.tolist()

        if not boxes:
            print('ERROR: No human detected')
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
    
    def odom_callback(self, msg : Odometry):
        self.odom_ = msg
    
    def map_callback(self, msg : OccupancyGrid):
        print('map data size =', len(msg.data))

    def drawMap(self, scan):

        map = OccupancyGrid()
        # Map header
        map.header.frame_id = 'base_link'
        map.header.stamp = self.get_clock().now().to_msg()
        # Map info
        map.info.resolution = self.SAFE_DISTANCE_ / 60
        map.info.height = 60
        map.info.width = 60
        map.info.origin.orientation = self.odom_.pose.pose.orientation
        map.info.origin.position.x = self.odom_.pose.pose.position.x + (map.info.resolution * map.info.width / 2)
        map.info.origin.position.y = self.odom_.pose.pose.position.y
        map.info.origin.position.z = self.odom_.pose.pose.position.z

        w = map.info.width
        h = map.info.height
        r = map.info.resolution
    
        # Map data
        i = 0
        map.data = []
        for i in range(3600):
            map.data.append(-1)
        map.data[10] = 100
        # print(len(map.data))
        # for j in range(h * w):
        #     for idx in range(360):
        #         if 0 < scan[idx][0] < 2 or -1 < scan[idx][1] < 1:
        # #             # if (scan[idx][1] > -1) and (scan[idx][1] < 1) and (scan[idx][0] < 2) and (scan[idx][0] > 0):
        # #             #     # if scan[idx][1] > (y * r) and scan[idx][1] < (r + (y * r)) and scan[idx][0] > (-1 + (x * r)) and scan[idx][0] < (-1 + r + (x * r)):
        #             map.data[j] = 100
        #             print(j, '=', map.data[j])

                        
        print(map.data)
        self.local_map_publisher_.publish(map)

    # def drawMap(self, scan):
    #     map = OccupancyGrid()
    #     # Map header
    #     map.header.frame_id = 'base_link'
    #     map.header.stamp = self.get_clock().now().to_msg()
    #     # Map info
    #     map.info.resolution = self.SAFE_DISTANCE_ / 60
    #     map.info.height = 60
    #     map.info.width = 60
    #     map.info.origin.orientation = self.odom_.pose.pose.orientation
    #     map.info.origin.position.x = self.odom_.pose.pose.position.x - (map.info.resolution * map.info.width / 2)
    #     map.info.origin.position.y = self.odom_.pose.pose.position.y - (map.info.resolution * map.info.height / 2)
    #     map.info.origin.position.z = self.odom_.pose.pose.position.z

    #     w = map.info.width
    #     h = map.info.height
    #     r = map.info.resolution

    #     # Map data
    #     map.data = [-1] * (w * h)  # заполняем массив значениями -1

    #     for i in range(len(scan)):
    #         x = scan[i][0]
    #         y = scan[i][1]
    #         if 0 <= x <= 2 and -1 <= y <= 1:
    #             map.data.append


    #     print(map.data)
    #     self.local_map_publisher_.publish(map)


    def scan_callback(self, msg : LaserScan):
        
        scan_ranges = msg.ranges
        angle_min = msg.angle_min
        inc = msg.angle_increment

        scan_points = []
        for idx in range (len(scan_ranges)):
            if str(scan_ranges[idx]) == 'inf':
                scan_points.append([100, 100])

            else:    
                angle = angle_min + idx * inc
                x = scan_ranges[idx] * np.cos(angle)
                y = scan_ranges[idx] * np.sin(angle)
                scan_points.append([x, y])
            # print(idx, '=', scan_points[idx])


        map = self.drawMap(scan_points)

    def occupancy_grid_callback(self, msg : OccupancyGrid):
        # Преобразование OccupancyGrid в массив numpy
        occupancy_grid_data = msg.data
        w = msg.info.width
        h = msg.info.height
        map_image = np.zeros((h, w), dtype='uint8')

        for y in range(h):
            for x in range(w):
                if occupancy_grid_data[y * w + x] == 100:
                    map_image[y][x] = 0
                if occupancy_grid_data[y * w + x] == 0:
                    map_image[y][x] = 255
                if occupancy_grid_data[y * w + x] == -1:
                    map_image[y][x] = 128                    


        # Преобразование массива numpy в изображение с помощью OpenCV
        map_image_np = cv2.resize(map_image, (w * 5, h * 5), interpolation=cv2.INTER_NEAREST)
        map_image_np = cv2.convertScaleAbs(map_image_np, alpha=(255.0/100.0))
        map_image_np = cv2.cvtColor(map_image_np, cv2.COLOR_GRAY2BGR)

        # Публикация изображения с помощью CvBridge
        image_msg = bridge.cv2_to_imgmsg(map_image_np, encoding="bgr8")
        # Публикация изображения
        self.map_img_publisher_.publish(image_msg)
        cv2.imshow('MAP', map_image_np)
        # cv2.circle(map_image_np, (10, 10), 5, (255, 255, 255))
        cv2.waitKey(1)    
      

def main(args=None):

    # Camera node
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
