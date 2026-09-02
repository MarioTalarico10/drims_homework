from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import (
    get_package_share_directory
)

import os


def generate_launch_description():

    package_dir = get_package_share_directory(
        'oak_dice_detector'
    )

    detector_config = os.path.join(
        package_dir,
        'config',
        'dice_detector.yaml'
    )

    calibration_file = os.path.join(
        package_dir,
        'config',
        'table_homography.yaml'
    )

    dice_detector_node = Node(
        package='oak_dice_detector',
        executable='dice_detector_node',
        name='oak_dice_detector',
        output='screen',

        parameters=[
            detector_config,

            {
                'calibration_file':
                    calibration_file
            }
        ]
    )

    return LaunchDescription([
        dice_detector_node
    ])