#!/usr/bin/env python3

import socket
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


class UDPForceReceiver(Node):

    def __init__(self):
        super().__init__('udp_force_receiver')

        # ROS parameters
        self.declare_parameter('udp_ip', '0.0.0.0')
        self.declare_parameter('udp_port', 5005)

        self.udp_ip = self.get_parameter(
            'udp_ip'
        ).get_parameter_value().string_value

        self.udp_port = self.get_parameter(
            'udp_port'
        ).get_parameter_value().integer_value

        # Publisher
        self.publisher_ = self.create_publisher(
            Vector3,
            '/force_d',
            10
        )

        # UDP socket
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

        self.sock.bind(
            (self.udp_ip, self.udp_port)
        )

        self.sock.setblocking(False)

        self.get_logger().info(
            f'Listening UDP on {self.udp_ip}:{self.udp_port}'
        )

        self.timer = self.create_timer(
            0.001,
            self.receive_udp
        )


    def receive_udp(self):

        try:
            data, addr = self.sock.recvfrom(65535)

        except BlockingIOError:
            return

        try:
            message = data.decode('utf-8')

            json_data = json.loads(message)

            fx = float(json_data['fx'])
            fy = float(json_data['fy'])
            fz = float(json_data['fz'])

            msg = Vector3()

            msg.x = fx
            msg.y = fy
            msg.z = fz

            self.publisher_.publish(msg)

        except (json.JSONDecodeError, KeyError, ValueError) as e:

            self.get_logger().warning(
                f'Invalid UDP packet: {e}'
            )


def main(args=None):

    rclpy.init(args=args)

    node = UDPForceReceiver()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.sock.close()
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()