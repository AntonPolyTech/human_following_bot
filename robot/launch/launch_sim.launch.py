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

    world = os.path.join(get_package_share_directory(package_name), 'worlds', 'cafe.world')
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
            '-x', '1.0',
            '-y', '-8.0',
            '-z', '1.0',
            '-R', '0.0',
            '-P', '0.0',
            '-Y', '1.57'
        ],
        output='screen',
        # namespace='master',
    )

    # spawn_entity_dyn = Node(
    #     package='gazebo_ros',
    #     executable='spawn_entity.py',
    #     arguments=[
    #         '-topic', 'robot_description',
    #         '-entity', 'dyn',
    #         '-x', '-0.34',
    #         '-y', '-0.18',
    #         '-z', '0.00'
    #     ],
    #     output='screen',
    #     namespace='dyn'
    # )

    goal_pose_pub = Node(
        package='graph_travel',
        executable='goal_pose_publisher',
        name='goal_pose_publisher',
    )

    joy_params = os.path.join(get_package_share_directory(
        'robot'), 'config', 'joystick.yaml')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        parameters=[joy_params],
    )

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_node',
        parameters=[joy_params],
        # remappings=[('/cmd_vel', 'diff_cont/cmd_vel_unstamped')],
    )

    rviz_file = os.path.join(get_package_share_directory(
        package_name), 'config', 'robot_rviz.rviz')

    rviz = Node(
        package='rviz2',
        namespace='',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file]
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(get_package_share_directory(
            'slam_toolbox'), 'launch', 'online_async_launch.py')]),
        launch_arguments={
            # 'params_file' : 'src/robot/config/mapper_params_online_async.yaml',
            'use_sim_time': 'true'

        }.items()
    )

    rtabmap = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        # parameters=[{
        #     'map_always_update': 'true',
        #     'frame_id': 'base_footprint',
        #     'use_sim_time': 'true'
        # }],
        remappings=[
            ('rgb/image', '/camera/image_raw'),
            ('rgb/camera_info', '/camera/camera_info'),
            ('depth/image', '/camera/depth/image_raw'),
        ],
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity_robot,
        # spawn_entity_dyn,
        # goal_pose_pub,
        # joy_node,
        # teleop_node,
        # rviz,
        # slam_toolbox,
        # rtabmap,
    ])
