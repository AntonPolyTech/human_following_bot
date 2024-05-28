# Requirements:
#   A realsense D435i
#   Install realsense2 ros2 package (ros-$ROS_DISTRO-realsense2-camera)
# Example:
#   $ ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true unite_imu_method:=1 enable_sync:=true align_depth.enable
#
#   $ ros2 launch rtabmap_examples realsense_d435i_color.launch.py
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    return LaunchDescription([

        # Nodes to launch       
        
        IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')]
        ),
        launch_arguments={'enable_gyro': 'true',
                          'enable_accel': 'true',
                          'unite_imu_method': '1',
                          'enable_sync': 'true',
                          'align_depth.enable': 'true'
                          }.items())
    ])
