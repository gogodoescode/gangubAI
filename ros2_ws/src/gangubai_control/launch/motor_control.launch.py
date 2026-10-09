"""
Launch motor controller with optional Gazebo simulation.

Usage:
  # Gazebo simulation:
  ros2 launch gangubai_control motor_control.launch.py simulate:=true wander_require_cliff_data:=false

  # Hardware mode (Raspberry Pi):
  ros2 launch gangubai_control motor_control.launch.py
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

try:
    get_package_share_directory('gazebo_ros')
    GAZEBO_AVAILABLE = True
except Exception:
    GAZEBO_AVAILABLE = False


def generate_launch_description():
    pkg_control = get_package_share_directory('gangubai_control')
    pkg_description = get_package_share_directory('gangubai_description')
    default_params = os.path.join(pkg_control, 'config', 'motor_params.yaml')
    default_world = os.path.join(pkg_description, 'worlds', 'cliff_test.world')

    # Read URDF for robot_state_publisher
    urdf_file = os.path.join(pkg_description, 'urdf', 'gangubai.urdf')
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    simulate_arg = DeclareLaunchArgument(
        'simulate', default_value='false',
        description='Launch Gazebo simulation (no GPIO)',
    )
    params_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to motor controller parameter file',
    )
    world_arg = DeclareLaunchArgument(
        'world', default_value=default_world,
        description='Gazebo world file to load',
    )
    wander_cliff_arg = DeclareLaunchArgument(
        'wander_require_cliff_data', default_value='true',
        description='Require cliff sensor data in wander mode for safety',
    )
    wander_cliff_topic_arg = DeclareLaunchArgument(
        'wander_cliff_topic', default_value='cliff_scan',
        description='Cliff topic for wander controller',
    )

    # ── Gazebo (only when simulate:=true) ────────────────────────────
    if GAZEBO_AVAILABLE:
        gazebo = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py',
                ])
            ),
            launch_arguments={'world': LaunchConfiguration('world')}.items(),
            condition=IfCondition(LaunchConfiguration('simulate')),
        )
    else:
        gazebo = None

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        condition=IfCondition(LaunchConfiguration('simulate')),
    )

    if GAZEBO_AVAILABLE:
        spawn_entity = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description',
                       '-entity', 'gangubai',
                       '-x', '0.0', '-y', '0.0', '-z', '1.0'],
            output='screen',
            condition=IfCondition(LaunchConfiguration('simulate')),
        )
    else:
        spawn_entity = None

    # ── Motor controller node ────────────────────────────────────────
    motor_controller = Node(
        package='gangubai_control',
        executable='motor_controller',
        name='motor_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'simulate': LaunchConfiguration('simulate')},
        ],
    )

    # ── Wander controller node (idles until /wander_mode=start) ─────
    wander_controller = Node(
        package='gangubai_control',
        executable='wander_controller',
        name='wander_controller',
        output='screen',
        parameters=[
            {
                'require_cliff_data': LaunchConfiguration('wander_require_cliff_data'),
                'cliff_topic': LaunchConfiguration('wander_cliff_topic'),
            },
        ],
    )

    actions = [
        simulate_arg, params_arg, world_arg,
        wander_cliff_arg, wander_cliff_topic_arg,
        motor_controller, wander_controller,
    ]
    if gazebo:
        actions.append(gazebo)
    actions.append(robot_state_publisher)
    if spawn_entity:
        actions.append(spawn_entity)
    return LaunchDescription(actions)
