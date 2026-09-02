from launch import LaunchDescription

from launch.actions import DeclareLaunchArgument

from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    udp_ip_arg = DeclareLaunchArgument(
        'udp_ip',
        default_value='0.0.0.0',
        description='IP address used to bind the UDP socket'
    )

    udp_port_arg = DeclareLaunchArgument(
        'udp_port',
        default_value='5005',
        description='UDP port used to receive force data'
    )

    udp_force_node = Node(
        package='udp_force_receiver',
        executable='udp_force_node',
        name='udp_force_receiver',
        output='screen',

        parameters=[
            {
                'udp_ip': LaunchConfiguration('udp_ip'),
                'udp_port': LaunchConfiguration('udp_port'),
            }
        ]
    )

    return LaunchDescription([
        udp_ip_arg,
        udp_port_arg,
        udp_force_node
    ])