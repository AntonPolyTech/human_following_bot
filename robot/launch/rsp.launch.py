import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

import xacro


def generate_launch_description():
    # Check if we're told to use sim time
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Process the URDF file
    pkg_path = os.path.join(get_package_share_directory('robot'))

    xacro_file_robot = os.path.join(pkg_path, 'description', 'robot.urdf.xacro')
    xacro_file_dyn = os.path.join(pkg_path, 'dyn_description', 'robot.urdf.xacro')

    robot_description_config_robot = xacro.process_file(xacro_file_robot)
    robot_description_config_dyn = xacro.process_file(xacro_file_dyn)

    # Create a robot_state_publisher node
    params_robot = {'robot_description': robot_description_config_robot.toxml(), 'use_sim_time': use_sim_time}
    params_dyn = {'robot_description': robot_description_config_dyn.toxml(), 'use_sim_time': use_sim_time}

    node_robot_state_publisher_robot = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params_robot],
        # namespace='master'
    )

    node_robot_state_publisher_dyn = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params_dyn],
        namespace='dyn'
    )

    # Launch!
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'),

        node_robot_state_publisher_robot,
        node_robot_state_publisher_dyn
    ])
