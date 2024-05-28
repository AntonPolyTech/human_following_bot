import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    package_name = 'robot'

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory(package_name), 'launch', 'rsp.launch.py')]
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    world = os.path.join(get_package_share_directory(package_name), 'worlds', 'static_test.world')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={'world': world}.items()
    )

    spawn_entity_robot = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'robot',
            '-x', '0.00',
            '-y', '-0.54',
            '-z', '0.00',
            '-Y', '-1.57',
        ],
        output='screen',
    )

    spawn_entity_dyn = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'dyn',
            '-x', '0.82',
            '-y', '-1.68',
            '-z', '0.00',
            '-Y', '3.14'
        ],
        output='screen',
        namespace='dyn'
    )

    rviz_file = os.path.join(get_package_share_directory(
        package_name), 'config', 'static_test.rviz')

    rviz = Node(
        package='rviz2',
        namespace='',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity_robot,
        spawn_entity_dyn,
        rviz,
    ])
