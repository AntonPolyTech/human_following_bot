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
from tf2_geometry_msgs import *
from tf2_ros import Time
from tf2_ros import TransformException
from .submodules.deep_sort.deep_sort.tracker import Tracker
from .submodules.sort import Sort


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
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

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
            Twist,
            '/cmd_vel',
            self.vel_callback,
            10)
        self.subscription3      
        
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
        # # For real robot
        self.image_sub_ = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub_ = message_filters.Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')
        
        # # For simulation
        # self.image_sub_ = message_filters.Subscriber(self, Image, '/camera/image_raw')
        # self.depth_sub_ = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        
        self.ts_ = message_filters.TimeSynchronizer([self.image_sub_, self.depth_sub_], 10)
        self.ts_.registerCallback(self.cam_callback)

        # Variables
        self.odom_ = Odometry()
        self.vel_ = Twist()
        self.mask_ = np.empty(0)
        self.P_3D_ = np.empty(0)
        self.bridge_ = CvBridge()
        self.tracker_ = Tracker()
        self.color_ = (128, 128, 128)
        self.SAFE_DISTANCE_ = 1.5 #meters
        # self.frame_ = cv2.namedWindow("frame")
        self.cursor_coords_ = np.empty(0)
        self.track_id_ = -1
        self.prev_track_id_ = []
        self.bbox_ = np.empty((0, 4), dtype=int)
        self.detection_threshold_ = 0.5
        self.prev_goal_pose_ = None

        self.sort_tracker_ = Sort(max_age=1200, min_hits=8, iou_threshold=0.15)


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
    
    # -------------------------------------------------------------------------------------- R G B P C 2   C A L L B A C K  

    def rgb_pc2_callback(self, _data):

        rgb = self.bridge_.imgmsg_to_cv2(_data.rgb, 'bgr8')

        # Getting human pixel coords
        tracker, self.mask_, P_2D = self.human_tracker(rgb)
        img_msg = self.bridge_.cv2_to_imgmsg(tracker)
        self.img_publisher_.publish(img_msg)

        cv2.imshow("frame", tracker)
        cv2.waitKey(1)

        # Setting mouse(left button) callback
        cv2.setMouseCallback("frame", self.mouse_callback)

        if self.tf_ == None:
            return

        if len(P_2D) == 0:
            print('P_2D is empty')
            return None

        # Getting target human 3D coords   
        xyz_array = pointcloud2_to_xyz_array(_data.pc2)
        self.P_3D_ = xyz_array[_data.rgb.width * P_2D[1] + P_2D[0]]

        humanDepth = self.humanDepth(xyz_array)

        if len(self.P_3D_) != 0:

            self.P_3D_[2] = self.findMedianValue(humanDepth)

            if self.P_3D_[2] <= self.SAFE_DISTANCE_:
                return None

            print('P_3D = ', self.P_3D_)
            
            # -------------------------------------------------------------------------------------- HUMAN POINT TEST 
            
            point = PointStamped()

            point.point.x = self.P_3D_[2]
            point.point.y = -self.P_3D_[0]
            point.point.z = self.odom_.pose.pose.position.z

            # global_point = do_transform_point(point, bl2map)

            point.header.frame_id = 'camera_link'
            point.header.stamp = self.get_clock().now().to_msg()
            
            self.point_publisher_.publish(point)
            # # # # # # # # # # # # # # # # # # # # # #
            
            # -------------------------------------------------------------------------------------- GOAL POSE
            goal_pose = PoseStamped()
            
            # Goal pose header
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Goal pose local coordinates
            goal_pose_local_coords = Point()
            
            x, y = self.calcRobotXY()

            goal_pose_local_coords.x = x
            goal_pose_local_coords.y = y
            goal_pose_local_coords.z = 0.0
            
            # Goal pose point for multiplication
            goal_pose_local_point = PointStamped()
            goal_pose_local_point.header.frame_id = 'camera_link'
            goal_pose_local_point.point = goal_pose_local_coords

            # Transform coordinates from base_link to map frame
            goal_pose_global_point = do_transform_point(goal_pose_local_point, self.tf_)
            target_real_coords = goal_pose_global_point.point

            # Goal pose position
            goal_pose.pose.position.x = target_real_coords.x
            goal_pose.pose.position.y = target_real_coords.y
            goal_pose.pose.position.z = target_real_coords.z

            # Goal pose orientation
            curr_robot_angles = self.getBaseLinkAngles()
            if len(curr_robot_angles) == 3:
                curr_robot_yaw = curr_robot_angles[2]

                q = quaternion_from_euler(0, 0, curr_robot_yaw - self.calcYaw())

                goal_pose.pose.orientation.x = q.x
                goal_pose.pose.orientation.y = q.y
                goal_pose.pose.orientation.z = q.z
                goal_pose.pose.orientation.w = q.w

                # if self.prev_goal_pose_ != None:
                #     goal_pose = self.GoalPoseFilter(self.prev_goal_pose_, goal_pose)
                #     self.updatePrevGoalPose()
                    # print('new id =', self.track_id_)
                    # print('prev id =', self.prev_track_id_)
                self.prev_goal_pose_ = goal_pose


                # if not self.goalPoseReached(self.tf_, goal_pose):
                self.goal_pose_publisher_.publish(goal_pose)
                

    # --------------------------------------------------------------------------------------C A M E R A   C A L L B A C K  

    def cam_callback(self, _rgb, _depth):
        rgb_depth = RgbDepth()
        rgb_depth.rgb = _rgb
        rgb_depth.depth = _depth
        self.rgb_depth_publisher_.publish(rgb_depth)
 
    def human_tracker(self, frame):
        # results = self.MODEL_.predict(source=frame, classes=0, save=False)
        # frame_ = results[0].plot()

        # boxes = results[0].boxes.xyxy.tolist()

        # if not boxes and not results[0].masks.data[0]:
        #     print("ERROR: No human detected")
        #     return frame, [], []
        
        # boxes = boxes[0]

        results = self.MODEL_.predict(source=frame, classes=0, save=False)

        if not results[0]:
            print("ERROR: No human detected")
            return frame, np.empty(0), np.empty(0)
        else:
                
            m = results[0].masks.data[0].cpu().numpy() * np.uint(255)

            # frame_ = results[0].plot()
            # cv2.imshow('mask', frame_)
            # cv2.waitKey(1)

            # if results[0].masks == None:
            #     return frame, [], []
            
            # if not results[0].boxes.xyxy.tolist() and not results[0].masks.data[0]:
            #     print("ERROR: No human detected")
            #     return frame, [], []

# ######################################## D E E P  S O R T

            # for result in results:
            #     detections = np.empty((0, 5), dtype=int) 
            #     for r in result.boxes.data.tolist():
            #         x1, y1, x2, y2, score, class_id = r
            #         x1 = int(x1)
            #         x2 = int(x2)
            #         y1 = int(y1)
            #         y2 = int(y2)
            #         class_id = int(class_id)
            #         # x1, y1, x2, y2, score, class_id = map(int, r)
            #         if score > self.detection_threshold_:
            #             detections = np.vstack([detections, [x1, y1, x2, y2, score]])

            #     self.tracker_.update(frame, detections)

            #     for track in self.tracker_.tracks:
            #         bbox = track.bbox
            #         x1, y1, x2, y2 = bbox
            #         track_id = track.track_id
            #         if track_id != self.track_id_:
            #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color_, 3)
            #             frame = cv2.putText(frame, f'track_id {track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_ , 3, cv2.LINE_AA)
            #         else:
            #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            #             frame = cv2.putText(frame, f'track_id {track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) , 3, cv2.LINE_AA)
            
            # self.getTrack()
######################################################

# ############################################# S O R T
        # results = self.predict(frame)

        # SORT
        # sort = Sort(max_age=20, min_hits=8, iou_threshold=0.15)

        detections_list = self.get_results(results)
        
        # SORT Tracking
        if len(detections_list) == 0:
            detections_list = np.empty((0, 5))

        res = self.sort_tracker_.update(detections_list)
            
        self.boxes_track_ = res
        print('REEEEEEEEEEEEEES', res)
        # self.boxes_track_ = res[:,:-1]
        # self.boxes_ids_ = res[:,-1].astype(int)

        
       
        # frame = self.draw_bounding_boxes_with_id(frame, self.boxes_track_, res[:,-1].astype(int))
        
        for track in self.boxes_track_:
            bbox = track[:-1]
            x1, y1, x2, y2 = bbox
            track_id = track[4]
            if track_id != self.track_id_:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color_, 3)
                frame = cv2.putText(frame, f'track_id {track_id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_ , 3, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                frame
        
        self.getTrack()
#############################################################

        print('BBOX =', self.bbox_)
        
        if len(self.bbox_) == 0:
            print("ERROR: No human detected")
            return frame, np.empty(0), np.empty(0)
        
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):  # Нажатие клавиши 'q' для выхода
        #     cv2.destroyAllWindows()
        
        # person coordinates
        X_p = int(self.bbox_[2] + self.bbox_[0]) // 2
        Y_p = int(self.bbox_[3] + self.bbox_[1]) // 2
        
        # 2D coordinates vector
        P_2D = np.array([X_p, Y_p, 1])
        print('Human pixel coordinates: ', P_2D)

        # drawing human body center 
        cv2.circle(frame, (X_p, Y_p), 5, (255, 0, 0), 10)
        
        return frame, m, P_2D
    
    def odom_callback(self, msg):
        self.odom_ = msg

    def vel_callback(self, msg):
        self.vel_ = msg

    def calcYaw(self):
        x = self.P_3D_[0]
        z = self.P_3D_[2]
        yaw_angle = np.arctan(x / z)  # radians
        return yaw_angle

    def getTransform(self):
        try:
            self.tf_ = self.tf_buffer.lookup_transform('map', 'camera_link', rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {1} to {1}: {ex}')
    
    # def mouse_callback(self, event, x, y, flags, param):
    #     self.prev_track_id_ = self.track_id_
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         self.cursor_coords_ = [x, y]
    #         tracks = self.track2array(self.tracker_.tracks)
    #         if len(self.cursor_coords_) == 2:
    #             for i in range(len(tracks)):
    #                 if (tracks[i][1] <= self.cursor_coords_[0] <= tracks[i][3]) and (tracks[i][2] <= self.cursor_coords_[1] <= tracks[i][4]):
    #                     self.track_id_ = tracks[i][0]
    #                     print("track_id", self.track_id_)

    def mouse_callback(self, event, x, y, flags, param):
        self.prev_track_id_ = self.track_id_
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cursor_coords_ = [x, y]
            tracks = self.boxes_track_
            if len(self.cursor_coords_) == 2:
                for i in range(len(tracks)):
                    if (tracks[i][0] <= self.cursor_coords_[0] <= tracks[i][2]) and (tracks[i][1] <= self.cursor_coords_[1] <= tracks[i][3]):
                        self.track_id_ = tracks[i][4]

    def track2array(self, tracks):
        track_array = np.empty((0, 5), dtype=np.float32)
        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox
            track_info = [track_id, bbox[0], bbox[1], bbox[2], bbox[3]]
            track_array = np.vstack([track_array, track_info]) 
        return track_array
    
    # def getTrack(self):  # deep sort
    #     self.bbox_ = np.empty(0)
    #     for track in self.track2array(self.tracker_.tracks):
    #         if track[0] == self.track_id_:
    #             self.bbox_ = track[1:]
    #         self.bbox_ = np.round(self.bbox_).astype(int)
    #         self.bbox_[self.bbox_ < 0] = 0

    def getTrack(self):  # sort
        self.bbox_ = np.empty(0)
        for track in self.boxes_track_:
            if track[4] == self.track_id_:
                self.bbox_ = track[:-1]
            self.bbox_ = np.round(self.bbox_).astype(int)
            self.bbox_[self.bbox_ < 0] = 0

    def goalPoseReached(self, curr_pose : TransformStamped, goal_pose : PoseStamped):
        
        dist = np.sqrt(np.square(goal_pose.pose.position.x) + np.square(goal_pose.pose.position.y)) - \
            np.sqrt(np.square(curr_pose.transform.translation.x) + np.square(curr_pose.transform.translation.y))
        
        print('DIST =', dist)

        if abs(dist) <= 0.35:
            return True
        else:
            return False

    def GoalPoseFilter(self, _prev_goal_pose : PoseStamped, _goal_pose : PoseStamped):
        if np.abs(_goal_pose.pose.position.y - _prev_goal_pose.pose.position.y) > 1.0:
        # dist = np.sqrt(np.square(_goal_pose.pose.position.x) + np.square(_goal_pose.pose.position.y)) - \
        #     np.sqrt(np.square(_prev_goal_pose.pose.position.x) + np.square(_prev_goal_pose.pose.position.y))
        # if dist > 4:
            return _prev_goal_pose
        return _goal_pose
    
    def updatePrevGoalPose(self):
        if self.prev_track_id_ != self.track_id_:
            self.prev_goal_pose_ = None        
    

    def predict(self, frame):
       
        results = self.MODEL_(frame, verbose=False)
        
        return results
    

    def get_results(self, results):
        
        detections_list = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            if class_id == 0:
                    
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()
                
                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]
                
                detections_list.append(merged_detection)
            
    
        return np.array(detections_list)
    
    
    def draw_bounding_boxes_with_id(self, img, bboxes, ids):
  
        for bbox, id_ in zip(bboxes, ids):

            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0,0,255),2)
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

            

def main(args=None):

    # Camera node
    rclpy.init(args=args)
    camera_subscriber = Camera_Subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
