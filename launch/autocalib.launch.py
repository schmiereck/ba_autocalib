"""Launch file for ba_autocalib_node.

Starts the calibration node with sensible defaults. The node assumes
MoveIt, the camera driver (CompressedImage + CameraInfo), and the TF
tree (robot_state_publisher) are already running — this launch file
only starts the calibration node itself.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ba_autocalib')
    default_perception_yaml = os.path.expanduser(
        '~/ros2_ws/src/ba_perception_pipeline/config/perception_pipeline.yaml')
    default_depth_yaml = os.path.expanduser(
        '~/ros2_ws/src/ba_perception_pipeline/config/depth_calibration.yaml')

    # venv handling (same pattern as ba_perception_pipeline.launch)
    venv_site = os.path.expanduser(
        '~/venvs/ba_depth_node/lib/python3.10/site-packages')
    env = os.environ.copy()
    if os.path.exists(venv_site):
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = (
            f'{venv_site}:{existing}' if existing else venv_site)

    marker_config = LaunchConfiguration('marker_config_file')
    calib_poses = LaunchConfiguration('calib_poses_file')
    perception_yaml = LaunchConfiguration('perception_yaml_path')
    depth_yaml = LaunchConfiguration('depth_yaml_path')

    return LaunchDescription([
        DeclareLaunchArgument(
            'marker_config_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'marker_config.yaml']),
        ),
        DeclareLaunchArgument(
            'calib_poses_file',
            default_value=PathJoinSubstitution(
                [pkg_share, 'config', 'calib_poses.yaml']),
        ),
        DeclareLaunchArgument(
            'perception_yaml_path',
            default_value=default_perception_yaml,
            description='Target file for hand-eye save'),
        DeclareLaunchArgument(
            'depth_yaml_path',
            default_value=default_depth_yaml,
            description='Target file for depth save'),
        Node(
            package='ba_autocalib',
            executable='autocalib_node',
            name='ba_autocalib_node',
            output='screen',
            parameters=[{
                'marker_config_file': marker_config,
                'calib_poses_file': calib_poses,
                'perception_yaml_path': perception_yaml,
                'depth_yaml_path': depth_yaml,
            }],
            env=env,
        ),
    ])
