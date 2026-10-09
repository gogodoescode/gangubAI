"""
Launch the motor + wander controllers, optionally inside Gazebo.

Usage:
  # Gazebo simulation:
  ros2 launch gangubai_control motor_control.launch.py simulate:=true wander_require_cliff_data:=false

  # Hardware mode (Raspberry Pi):
  ros2 launch gangubai_control motor_control.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('gangubai_control')
    with open(os.path.join(pkg, 'urdf', 'gangubai.urdf')) as f:
        robot_description = f.read()

    simulate = LaunchConfiguration('simulate')
    actions = [
        DeclareLaunchArgument(
            'simulate', default_value='false',
            description='Launch Gazebo simulation (no GPIO)',
        ),
        DeclareLaunchArgument(
            'wander_require_cliff_data', default_value='true',
            description='Require cliff sensor data in wander mode for safety',
        ),
        Node(
            package='gangubai_control',
            executable='motor_controller',
            name='motor_controller',
            output='screen',
            parameters=[{'simulate': simulate}],
        ),
        # Idles until /wander_mode receives "start".
        Node(
            package='gangubai_control',
            executable='wander_controller',
            name='wander_controller',
            output='screen',
            parameters=[{'require_cliff_data': LaunchConfiguration('wander_require_cliff_data')}],
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
            condition=IfCondition(simulate),
        ),
    ]

    # Gazebo pieces are only added when gazebo_ros is installed (not needed on the Pi).
    try:
        gazebo_ros = get_package_share_directory('gazebo_ros')
    except Exception:
        return LaunchDescription(actions)

    actions += [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_ros, 'launch', 'gazebo.launch.py')),
            launch_arguments={'world': os.path.join(pkg, 'worlds', 'cliff_test.world')}.items(),
            condition=IfCondition(simulate),
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description', '-entity', 'gangubai',
                       '-x', '0.0', '-y', '0.0', '-z', '1.0'],
            output='screen',
            condition=IfCondition(simulate),
        ),
    ]
    return LaunchDescription(actions)
