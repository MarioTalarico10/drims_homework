import cv2
import depthai as dai
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped, PoseStamped

from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point


class OakDiceDetector(Node):

    def __init__(self):

        super().__init__('oak_dice_detector')

        # ============================================================
        # ROS PARAMETERS
        # ============================================================

        self.declare_parameter(
            'camera_frame',
            'oak_rgb_camera_optical_frame'
        )

        # Frame in which the homography coordinates are expressed
        self.declare_parameter(
            'homography_frame',
            'camera_frame_front'
        )

        self.declare_parameter(
            'world_frame',
            'world'
        )

        self.declare_parameter(
            'calibration_file',
            ''
        )

        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('fps', 15.0)

        self.declare_parameter('show_preview', True)
        self.declare_parameter('show_mask', True)
        self.declare_parameter('show_face_roi', True)

        # ============================================================
        # YELLOW HSV SEGMENTATION
        # ============================================================

        self.declare_parameter('h_min', 18)
        self.declare_parameter('s_min', 80)
        self.declare_parameter('v_min', 80)

        self.declare_parameter('h_max', 40)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)

        # ============================================================
        # DICE DETECTION
        # ============================================================

        self.declare_parameter('min_dice_area', 300.0)
        self.declare_parameter('min_aspect_ratio', 0.55)
        self.declare_parameter('max_aspect_ratio', 1.60)

        # ============================================================
        # NORMALIZED FACE
        # ============================================================

        self.declare_parameter('face_roi_size', 200)
        self.declare_parameter('face_inner_margin', 0.12)

        # ============================================================
        # PIP FILTERING
        # ============================================================

        self.declare_parameter('pip_min_area_ratio', 0.002)
        self.declare_parameter('pip_max_area_ratio', 0.035)

        self.declare_parameter('pip_min_circularity', 0.45)

        self.declare_parameter('pip_min_radius_ratio', 0.025)
        self.declare_parameter('pip_max_radius_ratio', 0.13)

        self.declare_parameter('adaptive_block_size', 31)
        self.declare_parameter('adaptive_c', 8)

        self.declare_parameter('pip_close_kernel_size', 7)
        self.declare_parameter('pip_close_iterations', 1)

        self.declare_parameter('pip_open_kernel_size', 5)
        self.declare_parameter('pip_open_iterations', 1)

        # ============================================================
        # TEMPORAL FILTER
        # ============================================================

        self.declare_parameter('history_size', 7)
        self.declare_parameter('min_consistent_detections', 4)

        # ============================================================
        # READ BASIC PARAMETERS
        # ============================================================

        self.camera_frame = self.get_parameter(
            'camera_frame'
        ).value

        self.homography_frame = self.get_parameter(
            'homography_frame'
        ).value

        self.world_frame = self.get_parameter(
            'world_frame'
        ).value

        self.calibration_file = self.get_parameter(
            'calibration_file'
        ).value

        self.image_width = int(
            self.get_parameter(
                'image_width'
            ).value
        )

        self.image_height = int(
            self.get_parameter(
                'image_height'
            ).value
        )

        self.fps = float(
            self.get_parameter(
                'fps'
            ).value
        )

        self.show_preview = bool(
            self.get_parameter(
                'show_preview'
            ).value
        )

        self.show_mask = bool(
            self.get_parameter(
                'show_mask'
            ).value
        )

        self.show_face_roi = bool(
            self.get_parameter(
                'show_face_roi'
            ).value
        )

        self.face_history = []

        # ============================================================
        # LOAD CAMERA CALIBRATION + HOMOGRAPHY
        # ============================================================

        self.load_calibration(
            self.calibration_file
        )

        # ============================================================
        # UNDISTORTION MAPS
        #
        # Computed only once for efficiency.
        # ============================================================

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.camera_matrix,
            (
                self.image_width,
                self.image_height
            ),
            cv2.CV_32FC1
        )

        # ============================================================
        # TF2
        # ============================================================

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        )

        # ============================================================
        # ROS PUBLISHERS
        # ============================================================

        self.annotated_image_pub = self.create_publisher(
            Image,
            '/dice_camera/annotated_image',
            10
        )

        self.face_number_pub = self.create_publisher(
            Int32,
            '/dice_camera/face_number',
            10
        )

        self.pixel_center_pub = self.create_publisher(
            PointStamped,
            '/dice_camera/pixel_center',
            10
        )

        # Position resulting directly from homography
        self.dice_pose_camera_post_pub = self.create_publisher(
            PoseStamped,
            '/dice_camera/dice_pose_camera_front',
            10
        )

        # Same point transformed through TF into world
        self.dice_pose_world_pub = self.create_publisher(
            PoseStamped,
            '/dice_camera/dice_pose_world',
            10
        )

        self.bridge = CvBridge()

        # ============================================================
        # DEPTHAI V3
        # ============================================================

        self.pipeline = dai.Pipeline()

        self.camera = self.pipeline.create(
            dai.node.Camera
        )

        self.camera.build(
            dai.CameraBoardSocket.CAM_A
        )

        self.rgb_output = self.camera.requestOutput(
            size=(
                self.image_width,
                self.image_height
            ),
            type=dai.ImgFrame.Type.BGR888i
        )

        self.rgb_queue = (
            self.rgb_output.createOutputQueue(
                maxSize=1,
                blocking=False
            )
        )

        self.pipeline.start()

        # ============================================================
        # TIMER
        # ============================================================

        self.timer = self.create_timer(
            1.0 / self.fps,
            self.process_frame
        )

        self.get_logger().info(
            'OAK-D dice detector started'
        )

        self.get_logger().info(
            f'Homography frame: {self.homography_frame}'
        )

        self.get_logger().info(
            f'World frame: {self.world_frame}'
        )

        self.get_logger().info(
            f'Calibration file: {self.calibration_file}'
        )

    # ================================================================
    # LOAD CALIBRATION
    # ================================================================

    def load_calibration(self, filename):

        if not filename:

            raise RuntimeError(
                'calibration_file parameter is empty'
            )

        try:

            with open(filename, 'r') as file:

                data = yaml.safe_load(file)

        except Exception as ex:

            raise RuntimeError(
                f'Could not open calibration file '
                f'{filename}: {ex}'
            )

        required_keys = [
            'homography',
            'camera_matrix',
            'distortion_coefficients'
        ]

        for key in required_keys:

            if key not in data:

                raise RuntimeError(
                    f'Missing key "{key}" '
                    f'in calibration file'
                )

        self.H_homography = np.array(
            data['homography'],
            dtype=np.float64
        )

        self.camera_matrix = np.array(
            data['camera_matrix'],
            dtype=np.float64
        )

        self.dist_coeffs = np.array(
            data['distortion_coefficients'],
            dtype=np.float64
        ).reshape(-1)

        if self.H_homography.shape != (3, 3):

            raise RuntimeError(
                'Homography must be 3x3'
            )

        if self.camera_matrix.shape != (3, 3):

            raise RuntimeError(
                'Camera matrix must be 3x3'
            )

        self.get_logger().info(
            'Calibration successfully loaded'
        )

        self.get_logger().info(
            f'Homography:\n{self.H_homography}'
        )

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def process_frame(self):

        packet = self.rgb_queue.tryGet()

        if packet is None:
            return

        # ============================================================
        # RAW FRAME
        # ============================================================

        raw_frame = packet.getCvFrame()

        # ============================================================
        # UNDISTORT FRAME
        #
        # IMPORTANT:
        # the homography must have been computed using this same
        # undistorted image geometry.
        # ============================================================

        frame = cv2.remap(
            raw_frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR
        )

        annotated = frame.copy()

        # ============================================================
        # DICE DETECTION
        # ============================================================

        detection, mask = self.detect_dice(
            frame
        )

        if self.show_mask:

            cv2.imshow(
                'HSV Dice Mask',
                mask
            )

        # ============================================================
        # NO DICE
        # ============================================================

        if detection is None:

            self.face_history.clear()

            cv2.putText(
                annotated,
                'Dice: NOT DETECTED',
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            self.publish_annotated_image(
                annotated
            )

            self.display_preview(
                annotated
            )

            return

        contour = detection['contour']

        x = detection['x']
        y = detection['y']
        w = detection['w']
        h = detection['h']

        u = detection['u']
        v = detection['v']

        # ============================================================
        # PIXEL -> CAMERA_POST_FRONT
        # ============================================================

        x_h, y_h = self.pixel_to_homography_frame(
            u,
            v
        )

        # ============================================================
        # PUBLISH CAMERA_POST + WORLD POSES
        # ============================================================

        world_position = self.publish_dice_poses(
            x_h,
            y_h
        )

        # ============================================================
        # TOP FACE
        # ============================================================

        face_roi = self.extract_top_face(
            frame,
            contour
        )

        raw_number = None
        pip_centers = []

        if face_roi is not None:

            raw_number, pip_centers, binary_pips = (
                self.count_pips(
                    face_roi
                )
            )

            if self.show_face_roi:

                face_display = face_roi.copy()

                for px, py in pip_centers:

                    cv2.circle(
                        face_display,
                        (px, py),
                        7,
                        (0, 0, 255),
                        2
                    )

                cv2.imshow(
                    'Normalized Dice Face',
                    face_display
                )

                cv2.imshow(
                    'Pip Binary Mask',
                    binary_pips
                )

        # ============================================================
        # TEMPORAL FILTER
        # ============================================================

        stable_number = self.temporal_filter(
            raw_number
        )

        # ============================================================
        # BOUNDING BOX
        # ============================================================

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # ============================================================
        # TEXT ANNOTATIONS
        # ============================================================

        cv2.putText(
            annotated,
            f'Center: ({u}, {v}) px',
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            (
                f'{self.homography_frame}: '
                f'X={x_h:.3f} '
                f'Y={y_h:.3f} m'
            ),
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if world_position is not None:

            x_world, y_world, z_world = (
                world_position
            )

            cv2.putText(
                annotated,
                (
                    f'World: '
                    f'X={x_world:.3f} '
                    f'Y={y_world:.3f} '
                    f'Z={z_world:.3f} m'
                ),
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ============================================================
        # FACE NUMBER
        # ============================================================

        if stable_number is not None:

            cv2.putText(
                annotated,
                f'Dice face: {stable_number}',
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            self.publish_face_number(
                stable_number
            )

        elif raw_number is not None:

            cv2.putText(
                annotated,
                f'Dice face: {raw_number} (unstable)',
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                annotated,
                'Dice face: UNKNOWN',
                (30, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # ============================================================
        # ROS OUTPUT
        # ============================================================

        self.publish_pixel_center(
            u,
            v
        )

        self.publish_annotated_image(
            annotated
        )

        self.display_preview(
            annotated
        )

    # ================================================================
    # PIXEL -> HOMOGRAPHY FRAME
    # ================================================================

    def pixel_to_homography_frame(
        self,
        u,
        v
    ):

        pixel = np.array(
            [[[float(u), float(v)]]],
            dtype=np.float32
        )

        metric = cv2.perspectiveTransform(
            pixel,
            self.H_homography
        )

        x = float(
            metric[0, 0, 0]
        )

        y = float(
            metric[0, 0, 1]
        )

        return x, y

    # ================================================================
    # PUBLISH BOTH POSES
    # ================================================================

    def publish_dice_poses(
        self,
        x_h,
        y_h
    ):

        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # ============================================================
        # POSE IN camera_post_front
        # ============================================================

        pose_h = PoseStamped()

        pose_h.header.stamp = stamp
        pose_h.header.frame_id = (
            self.homography_frame
        )

        pose_h.pose.position.x = float(x_h)
        pose_h.pose.position.y = float(y_h)
        pose_h.pose.position.z = 0.0

        # Orientation is NOT estimated.
        pose_h.pose.orientation.x = 0.0
        pose_h.pose.orientation.y = 0.0
        pose_h.pose.orientation.z = 0.0
        pose_h.pose.orientation.w = 1.0

        self.dice_pose_camera_post_pub.publish(
            pose_h
        )

        # ============================================================
        # LOOKUP TF camera_post_front -> world
        # ============================================================

        try:

            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.homography_frame,
                Time(),
                timeout=Duration(
                    seconds=0.1
                )
            )

        except TransformException as ex:

            self.get_logger().warn(
                f'Cannot transform '
                f'{self.homography_frame} -> '
                f'{self.world_frame}: {ex}'
            )

            return None

        # ============================================================
        # TRANSFORM POINT
        # ============================================================

        point_h = PointStamped()

        point_h.header = pose_h.header

        point_h.point.x = pose_h.pose.position.x
        point_h.point.y = pose_h.pose.position.y
        point_h.point.z = pose_h.pose.position.z

        point_world = do_transform_point(
            point_h,
            transform
        )

        # ============================================================
        # WORLD POSE
        # ============================================================

        pose_world = PoseStamped()

        pose_world.header.stamp = stamp
        pose_world.header.frame_id = (
            self.world_frame
        )

        pose_world.pose.position.x = (
            point_world.point.x
        )

        pose_world.pose.position.y = (
            point_world.point.y
        )

        pose_world.pose.position.z = (
            point_world.point.z
        )

        # Again, orientation is NOT estimated.
        pose_world.pose.orientation.x = 0.0
        pose_world.pose.orientation.y = 0.0
        pose_world.pose.orientation.z = 0.0
        pose_world.pose.orientation.w = 1.0

        self.dice_pose_world_pub.publish(
            pose_world
        )

        return (
            point_world.point.x,
            point_world.point.y,
            point_world.point.z
        )

    # ================================================================
    # DICE SEGMENTATION
    # ================================================================

    def detect_dice(
        self,
        frame
    ):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        lower = np.array(
            [
                self.get_parameter('h_min').value,
                self.get_parameter('s_min').value,
                self.get_parameter('v_min').value
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                self.get_parameter('h_max').value,
                self.get_parameter('s_max').value,
                self.get_parameter('v_max').value
            ],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower,
            upper
        )

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None, mask

        min_area = float(
            self.get_parameter(
                'min_dice_area'
            ).value
        )

        min_ratio = float(
            self.get_parameter(
                'min_aspect_ratio'
            ).value
        )

        max_ratio = float(
            self.get_parameter(
                'max_aspect_ratio'
            ).value
        )

        valid = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            if h == 0:
                continue

            ratio = w / float(h)

            if not (
                min_ratio <= ratio <= max_ratio
            ):
                continue

            valid.append(
                contour
            )

        if not valid:
            return None, mask

        contour = max(
            valid,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        M = cv2.moments(
            contour
        )

        if M['m00'] == 0:
            return None, mask

        u = int(
            M['m10']
            /
            M['m00']
        )

        v = int(
            M['m01']
            /
            M['m00']
        )

        detection = {
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'u': u,
            'v': v
        }

        return detection, mask

    # ================================================================
    # NORMALIZE TOP FACE
    # ================================================================

    def extract_top_face(
        self,
        frame,
        contour
    ):

        rect = cv2.minAreaRect(
            contour
        )

        box = cv2.boxPoints(
            rect
        )

        box = np.asarray(
            box,
            dtype=np.float32
        )

        box = self.order_points(
            box
        )

        size = int(
            self.get_parameter(
                'face_roi_size'
            ).value
        )

        destination = np.array(
            [
                [0, 0],
                [size - 1, 0],
                [size - 1, size - 1],
                [0, size - 1]
            ],
            dtype=np.float32
        )

        H = cv2.getPerspectiveTransform(
            box,
            destination
        )

        warped = cv2.warpPerspective(
            frame,
            H,
            (size, size)
        )

        margin_ratio = float(
            self.get_parameter(
                'face_inner_margin'
            ).value
        )

        margin = int(
            size * margin_ratio
        )

        if margin * 2 >= size:
            return warped

        inner = warped[
            margin:size - margin,
            margin:size - margin
        ]

        inner = cv2.resize(
            inner,
            (size, size),
            interpolation=cv2.INTER_CUBIC
        )

        return inner

    # ================================================================
    # ORDER QUADRILATERAL
    # ================================================================

    def order_points(
        self,
        pts
    ):

        ordered = np.zeros(
            (4, 2),
            dtype=np.float32
        )

        sums = pts.sum(
            axis=1
        )

        differences = np.diff(
            pts,
            axis=1
        ).reshape(-1)

        ordered[0] = pts[
            np.argmin(sums)
        ]

        ordered[2] = pts[
            np.argmax(sums)
        ]

        ordered[1] = pts[
            np.argmin(differences)
        ]

        ordered[3] = pts[
            np.argmax(differences)
        ]

        return ordered

    # ================================================================
    # PIP DETECTION
    # ================================================================

    def count_pips(
        self,
        face_roi
    ):

        gray = cv2.cvtColor(
            face_roi,
            cv2.COLOR_BGR2GRAY
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(
            gray
        )

        gray = cv2.medianBlur(
            gray,
            3
        )

        block_size = int(
            self.get_parameter(
                'adaptive_block_size'
            ).value
        )

        if block_size % 2 == 0:
            block_size += 1

        if block_size < 3:
            block_size = 3

        adaptive_c = float(
            self.get_parameter(
                'adaptive_c'
            ).value
        )

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            adaptive_c
        )

        # ============================================================
        # MORPHOLOGY PARAMETERS
        # ============================================================

        close_size = int(
            self.get_parameter(
                'pip_close_kernel_size'
            ).value
        )

        close_iterations = int(
            self.get_parameter(
                'pip_close_iterations'
            ).value
        )

        open_size = int(
            self.get_parameter(
                'pip_open_kernel_size'
            ).value
        )

        open_iterations = int(
            self.get_parameter(
                'pip_open_iterations'
            ).value
        )

        if close_size % 2 == 0:
            close_size += 1

        if open_size % 2 == 0:
            open_size += 1

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size)
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=open_iterations
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = binary.shape

        total_area = float(
            h * w
        )

        min_area_ratio = float(
            self.get_parameter(
                'pip_min_area_ratio'
            ).value
        )

        max_area_ratio = float(
            self.get_parameter(
                'pip_max_area_ratio'
            ).value
        )

        min_circularity = float(
            self.get_parameter(
                'pip_min_circularity'
            ).value
        )

        min_radius_ratio = float(
            self.get_parameter(
                'pip_min_radius_ratio'
            ).value
        )

        max_radius_ratio = float(
            self.get_parameter(
                'pip_max_radius_ratio'
            ).value
        )

        image_scale = float(
            min(h, w)
        )

        pip_centers = []

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            area_ratio = (
                area
                /
                total_area
            )

            if not (
                min_area_ratio
                <= area_ratio
                <= max_area_ratio
            ):
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            circularity = (
                4.0
                *
                np.pi
                *
                area
                /
                (
                    perimeter
                    *
                    perimeter
                )
            )

            if circularity < min_circularity:
                continue

            _, radius = cv2.minEnclosingCircle(
                contour
            )

            radius_ratio = (
                radius
                /
                image_scale
            )

            if not (
                min_radius_ratio
                <= radius_ratio
                <= max_radius_ratio
            ):
                continue

            M = cv2.moments(
                contour
            )

            if M['m00'] == 0:
                continue

            cx = int(
                M['m10']
                /
                M['m00']
            )

            cy = int(
                M['m01']
                /
                M['m00']
            )

            border = int(
                0.08
                *
                min(h, w)
            )

            if (
                cx < border
                or
                cy < border
                or
                cx >= w - border
                or
                cy >= h - border
            ):
                continue

            pip_centers.append(
                (
                    cx,
                    cy
                )
            )

        pip_centers = self.remove_duplicate_points(
            pip_centers,
            minimum_distance=15
        )

        number = len(
            pip_centers
        )

        if number < 1 or number > 6:

            return (
                None,
                pip_centers,
                binary
            )

        if not self.validate_dice_pattern(
            pip_centers,
            number,
            w,
            h
        ):

            return (
                None,
                pip_centers,
                binary
            )

        return (
            number,
            pip_centers,
            binary
        )

    # ================================================================
    # REMOVE DUPLICATE POINTS
    # ================================================================

    def remove_duplicate_points(
        self,
        points,
        minimum_distance
    ):

        filtered = []

        for point in points:

            keep = True

            for existing in filtered:

                distance = np.linalg.norm(
                    np.array(point)
                    -
                    np.array(existing)
                )

                if distance < minimum_distance:

                    keep = False
                    break

            if keep:

                filtered.append(
                    point
                )

        return filtered

    # ================================================================
    # VALIDATE DICE PATTERN
    # ================================================================

    def validate_dice_pattern(
        self,
        points,
        number,
        width,
        height
    ):

        if number == 1:

            x, y = points[0]

            center_x = width / 2.0
            center_y = height / 2.0

            max_distance = (
                0.25
                *
                min(
                    width,
                    height
                )
            )

            distance = np.hypot(
                x - center_x,
                y - center_y
            )

            return (
                distance
                <
                max_distance
            )

        pts = np.array(
            points,
            dtype=np.float32
        )

        x_range = (
            pts[:, 0].max()
            -
            pts[:, 0].min()
        )

        y_range = (
            pts[:, 1].max()
            -
            pts[:, 1].min()
        )

        if number >= 4:

            if x_range < 0.25 * width:
                return False

            if y_range < 0.25 * height:
                return False

        return True

    # ================================================================
    # TEMPORAL FILTER
    # ================================================================

    def temporal_filter(
        self,
        raw_number
    ):

        history_size = int(
            self.get_parameter(
                'history_size'
            ).value
        )

        min_consistent = int(
            self.get_parameter(
                'min_consistent_detections'
            ).value
        )

        self.face_history.append(
            raw_number
        )

        if (
            len(self.face_history)
            >
            history_size
        ):

            self.face_history.pop(
                0
            )

        valid_values = [
            n
            for n in self.face_history
            if n is not None
        ]

        if len(valid_values) < min_consistent:
            return None

        values, counts = np.unique(
            valid_values,
            return_counts=True
        )

        best_index = np.argmax(
            counts
        )

        best_value = int(
            values[
                best_index
            ]
        )

        best_count = int(
            counts[
                best_index
            ]
        )

        if best_count < min_consistent:
            return None

        return best_value

    # ================================================================
    # PREVIEW
    # ================================================================

    def display_preview(
        self,
        frame
    ):

        if self.show_preview:

            cv2.imshow(
                'OAK-D Dice Detector',
                frame
            )

        if (
            self.show_preview
            or self.show_mask
            or self.show_face_roi
        ):

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):

                self.get_logger().info(
                    'Closing OpenCV windows'
                )

                self.show_preview = False
                self.show_mask = False
                self.show_face_roi = False

                cv2.destroyAllWindows()

    # ================================================================
    # ROS PUBLICATION
    # ================================================================

    def publish_face_number(
        self,
        number
    ):

        msg = Int32()

        msg.data = int(number)

        self.face_number_pub.publish(
            msg
        )

    def publish_pixel_center(
        self,
        u,
        v
    ):

        msg = PointStamped()

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # NOTE:
        # x and y here are PIXELS, not metric coordinates.
        msg.header.frame_id = (
            self.camera_frame
        )

        msg.point.x = float(u)
        msg.point.y = float(v)
        msg.point.z = 0.0

        self.pixel_center_pub.publish(
            msg
        )

    def publish_annotated_image(
        self,
        frame
    ):

        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        msg.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        msg.header.frame_id = (
            self.camera_frame
        )

        self.annotated_image_pub.publish(
            msg
        )

    # ================================================================
    # CLEANUP
    # ================================================================

    def destroy_node(self):

        cv2.destroyAllWindows()

        if hasattr(
            self,
            'pipeline'
        ):

            self.pipeline.stop()

        super().destroy_node()


def main(args=None):

    rclpy.init(
        args=args
    )

    node = OakDiceDetector()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == '__main__':
    main()