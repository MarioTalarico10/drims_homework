#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from dice_interfaces.srv import DetectDice


class DiceDetectorService(Node):

    def __init__(self):
        super().__init__('dice_detector_service')

        # ============================================================
        # CAMERA / SERVICE PARAMETERS
        # ============================================================
        self.declare_parameter('image_width', 1280)
        self.declare_parameter('image_height', 720)
        self.declare_parameter('dice_z', 0.585)
        self.declare_parameter('undistort_image', True)
        self.declare_parameter('world_frame', 'base_link')
        self.declare_parameter('camera_link_frame', 'camera_link')
        self.declare_parameter('dice_frame', 'dice_frame')

        # ============================================================
        # DICE SEGMENTATION - SAME TUNING AS FINAL NODE
        # ============================================================
        self.declare_parameter('h_min', 23)
        self.declare_parameter('s_min', 150)
        self.declare_parameter('v_min', 75)
        self.declare_parameter('h_max', 33)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)
        self.declare_parameter('min_dice_area', 250.0)
        self.declare_parameter('min_aspect_ratio', 0.45)
        self.declare_parameter('max_aspect_ratio', 1.90)
        self.declare_parameter('dice_close_ratio', 0.035)
        self.declare_parameter('dice_open_ratio', 0.015)

        # White/specular pips fallback
        self.declare_parameter('white_pip_v_min', 200)
        self.declare_parameter('white_pip_s_max', 100)

        # ============================================================
        # RAW PIP DETECTION INSIDE COMPLETE DICE
        # ============================================================
        self.declare_parameter('raw_pip_gray_threshold', 40)
        self.declare_parameter('raw_pip_min_area_ratio', 0.0010)
        self.declare_parameter('raw_pip_max_area_ratio', 0.0750)
        self.declare_parameter('raw_pip_min_circularity', 0.25)
        self.declare_parameter('raw_pip_max_aspect_ratio', 2.30)

        # ============================================================
        # TOP FACE / ELLIPSE MODEL
        # ============================================================
        self.declare_parameter('top_pip_radial_gate', 0.4)
        self.declare_parameter('top_face_box_scale', 1.35)
        self.declare_parameter('top_face_min_size_ratio', 0.60)
        self.declare_parameter('top_face_max_size_ratio', 1.00)
        self.declare_parameter('ellipse_perspective_strength', 0.30)

        # ============================================================
        # RECTIFICATION / FINAL PIP DETECTION
        # ============================================================
        self.declare_parameter('rectified_size', 300)
        self.declare_parameter('pip_gray_threshold', 40)
        self.declare_parameter('pip_min_area_ratio', 0.0015)
        self.declare_parameter('pip_max_area_ratio', 0.08)
        self.declare_parameter('pip_min_circularity', 0.35)
        self.declare_parameter('pip_border_margin_ratio', 0.015)

        # ============================================================
        # READ PARAMETERS
        # ============================================================
        self.image_width = int(self.get_parameter('image_width').value)
        self.image_height = int(self.get_parameter('image_height').value)
        self.dice_z = float(self.get_parameter('dice_z').value)
        self.use_undistortion = bool(self.get_parameter('undistort_image').value)
        self.rectified_size = int(self.get_parameter('rectified_size').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.camera_link_frame = str(self.get_parameter('camera_link_frame').value)
        self.dice_frame = str(self.get_parameter('dice_frame').value)

        # ============================================================
        # CAMERA CALIBRATION, READ DIRECTLY FROM OAK EEPROM
        # ============================================================
        self.camera_matrix = None
        self.dist_coeffs = None
        self.processing_camera_matrix = None
        self.processing_image_size = None

        # ============================================================
        # OPTIONAL DEBUG FILES - SAME NAMES AS FINAL NODE
        # ============================================================
        self.output_image_path = '/home/drims/bags/dice_detected.png'
        self.output_mask_path = '/home/drims/bags/dice_mask.png'
        self.output_top_face_mask_path = '/home/drims/bags/top_face_mask.png'
        self.output_rectified_path = '/home/drims/bags/dice_rectified.png'
        self.output_pip_mask_path = '/home/drims/bags/pip_mask.png'
        self.output_raw_pip_mask_path = '/home/drims/bags/raw_pip_mask.png'

        # ============================================================
        # TF / OPTIONAL TOPIC OUTPUTS - KEPT FOR DEBUG COMPATIBILITY
        # ============================================================
        self.position_pub = self.create_publisher(
            PointStamped,
            '/dice_detector/position_camera',
            10
        )
        self.pose_camera_pub = self.create_publisher(
            PoseStamped,
            '/dice_detector/pose_camera',
            10
        )
        self.pose_world_pub = self.create_publisher(
            PoseStamped,
            '/dice_detector/pose_world',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.dice_tf_broadcaster = TransformBroadcaster(self)
        self.last_dice_transform = None
        self.tf_timer = self.create_timer(0.1, self.republish_dice_tf)

        # ============================================================
        # OAK DEVICE
        # ============================================================
        self.device = None
        self.rgb_queue = None
        self.initialize_oak_camera()

        # ============================================================
        # ROS SERVICE
        # ============================================================
        self.service = self.create_service(
            DetectDice,
            '/detect_dice',
            self.detect_dice_service_callback
        )

        self.get_logger().info('Dice service ready on /detect_dice')
        self.get_logger().info(
            f'OAK RGB output: {self.image_width}x{self.image_height}'
        )
        self.get_logger().info(f'World frame: {self.world_frame}')
        self.get_logger().info(f'Camera frame: {self.camera_link_frame}')

    # ================================================================
    # OAK INITIALIZATION + EEPROM CALIBRATION
    # ================================================================
    def initialize_oak_camera(self):
        self.get_logger().info('Initializing physical OAK RGB camera...')

        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName('rgb')

        cam_rgb.setPreviewSize(self.image_width, self.image_height)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.preview.link(xout_rgb.input)

        try:
            self.device = dai.Device(pipeline)
        except Exception as exc:
            raise RuntimeError(f'Could not open OAK device: {exc}') from exc

        calibration = self.device.readCalibration()
        rgb_socket = dai.CameraBoardSocket.CAM_A

        intrinsics = calibration.getCameraIntrinsics(
            rgb_socket,
            self.image_width,
            self.image_height
        )
        self.camera_matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)

        distortion = calibration.getDistortionCoefficients(rgb_socket)
        self.dist_coeffs = np.asarray(distortion, dtype=np.float64).reshape(-1)

        self.rgb_queue = self.device.getOutputQueue(
            name='rgb',
            maxSize=2,
            blocking=True
        )

        self.get_logger().info('OAK camera opened successfully.')
        self.get_logger().info(f'EEPROM camera matrix:\n{self.camera_matrix}')
        self.get_logger().info(
            f'EEPROM distortion coefficients: {self.dist_coeffs}'
        )

    # ================================================================
    # ACQUIRE FRESH RGB FRAME
    # ================================================================
    def acquire_rgb_frame(self):
        if self.rgb_queue is None:
            return None

        # Empty queued stale frames when possible, then wait for a fresh one.
        latest = None
        try:
            while self.rgb_queue.has():
                latest = self.rgb_queue.get()
        except Exception:
            latest = None

        packet = self.rgb_queue.get()
        if packet is None:
            return None

        frame = packet.getCvFrame()
        if frame is None:
            return None

        return frame

    # ================================================================
    # UNDISTORT USING OAK EEPROM CALIBRATION
    # ================================================================
    def undistort_frame(self, frame):
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError('OAK camera calibration is not available.')

        if not self.use_undistortion:
            self.processing_camera_matrix = self.camera_matrix.copy()
            self.processing_image_size = (frame.shape[1], frame.shape[0])
            return frame.copy()

        h, w = frame.shape[:2]
        current_size = (w, h)

        if (
            self.processing_camera_matrix is None
            or self.processing_image_size != current_size
        ):
            self.processing_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                current_size,
                1.0,
                current_size
            )
            self.processing_image_size = current_size
            self.get_logger().info(
                f'Processing camera matrix:\n{self.processing_camera_matrix}'
            )

        return cv2.undistort(
            frame,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.processing_camera_matrix
        )

    # ================================================================
    # OPENCV -> sensor_msgs/CompressedImage
    # ================================================================
    def cv_to_compressed_image(self, frame, stamp):
        ok, encoded = cv2.imencode(
            '.jpg',
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError('Could not JPEG-encode annotated image.')

        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_link_frame
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()
        return msg

    # ================================================================
    # SERVICE CALLBACK
    # ================================================================
    def detect_dice_service_callback(self, request, response):
        del request
        self.get_logger().info('/detect_dice called: acquiring OAK frame...')

        response.success = False
        response.face_value = 0
        response.message = ''
        response.pose_world.header.frame_id = self.world_frame

        try:
            frame_raw = self.acquire_rgb_frame()
            if frame_raw is None:
                response.message = 'Could not acquire RGB frame from OAK.'
                return response

            stamp = self.get_clock().now().to_msg()
            result = self.process_frame(frame_raw, stamp)

            # Always return an image when acquisition succeeded, even if the die
            # was not detected. This makes service-side debugging much easier.
            response.annotated_image = self.cv_to_compressed_image(
                result['annotated'],
                stamp
            )

            if not result['success']:
                response.message = result['message']
                return response

            if result['pose_world'] is None:
                response.message = (
                    'Dice detected and pips counted, but world pose is unavailable. '
                    'Check TF world <- camera_link.'
                )
                return response

            response.success = True
            response.pose_world = result['pose_world']
            response.face_value = int(result['pip_count'])
            response.message = 'Dice detected successfully.'

            self.get_logger().info(
                f"Service result: face={result['pip_count']}, "
                f"world=({result['pose_world'].pose.position.x:.4f}, "
                f"{result['pose_world'].pose.position.y:.4f}, "
                f"{result['pose_world'].pose.position.z:.4f})"
            )
            return response

        except Exception as exc:
            self.get_logger().error(f'DetectDice service failed: {exc}')
            response.success = False
            response.face_value = 0
            response.message = f'Service error: {exc}'
            return response

    # ================================================================
    # COMPLETE ONE-FRAME DETECTION PIPELINE
    # ================================================================
    def process_frame(self, frame_raw, stamp):
        frame = self.undistort_frame(frame_raw)
        annotated = frame.copy()

        detection, dice_mask = self.detect_dice(frame)

        rectified_debug = None
        pip_mask = None
        top_face_mask = None
        raw_pip_mask = None

        result = {
            'success': False,
            'message': 'Dice not detected.',
            'annotated': annotated,
            'pose_camera': None,
            'pose_world': None,
            'pip_count': 0,
            'yaw_world': None,
        }

        if detection is None:
            self.get_logger().warn('Dice NOT detected.')
            cv2.putText(
                annotated,
                'Dice: NOT DETECTED',
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
            cv2.imwrite(self.output_image_path, annotated)
            cv2.imwrite(self.output_mask_path, dice_mask)
            return result

        contour = detection['contour']
        dice_rect = detection['rect']
        u = detection['u']
        v = detection['v']

        self.draw_dice_oriented_box(annotated, dice_rect)

        raw_candidates, raw_pip_mask = self.detect_raw_pips(
            frame,
            contour,
            dice_rect
        )

        top_face_box, ellipse = self.estimate_top_face_from_pips(
            frame,
            raw_candidates,
            dice_rect,
            (u, v)
        )
        del ellipse

        if top_face_box is not None:
            top_face_box_int = np.round(top_face_box).astype(np.int32)
            top_face_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(top_face_mask, top_face_box_int, 255)
            rectified, H = self.rectify_dice_face(frame, top_face_box)
            del H
        else:
            rectified = None
            self.get_logger().warn('Could not estimate top face from pips.')

        if rectified is not None:
            pip_count, pip_centers, pip_mask = self.count_pips(rectified)
            rectified_debug = rectified.copy()
            for cx, cy in pip_centers:
                cv2.circle(rectified_debug, (cx, cy), 10, (0, 0, 255), 2)
            cv2.putText(
                rectified_debug,
                f'Value = {pip_count}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        else:
            pip_count = 0

        position = self.pixel_to_camera_position(u, v)
        pose_camera = None
        pose_world = None
        yaw_world = None

        if position is not None:
            X, Y, Z = position
            self.publish_position(X, Y, Z, stamp)

            pose_camera, pose_world, yaw_world = self.estimate_dice_poses(
                X,
                Y,
                Z,
                dice_rect,
                stamp
            )

            if pose_camera is not None:
                self.pose_camera_pub.publish(pose_camera)

            if pose_world is not None:
                self.pose_world_pub.publish(pose_world)
                self.broadcast_dice_tf(pose_world)

            cv2.putText(
                annotated,
                f'X = {X:.3f} m',
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                f'Y = {Y:.3f} m',
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated,
                f'Z = {Z:.3f} m',
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.putText(
            annotated,
            f'Dice value = {pip_count}',
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        # Preserve the same debug outputs used during tuning.
        cv2.imwrite(self.output_image_path, annotated)
        cv2.imwrite(self.output_mask_path, dice_mask)
        if top_face_mask is not None:
            cv2.imwrite(self.output_top_face_mask_path, top_face_mask)
        if rectified_debug is not None:
            cv2.imwrite(self.output_rectified_path, rectified_debug)
        if pip_mask is not None:
            cv2.imwrite(self.output_pip_mask_path, pip_mask)
        if raw_pip_mask is not None:
            cv2.imwrite(self.output_raw_pip_mask_path, raw_pip_mask)

        result.update({
            'success': True,
            'message': 'Dice detected.',
            'annotated': annotated,
            'pose_camera': pose_camera,
            'pose_world': pose_world,
            'pip_count': int(pip_count),
            'yaw_world': yaw_world,
        })
        return result

    def detect_dice(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array(
            [
                int(self.get_parameter('h_min').value),
                int(self.get_parameter('s_min').value),
                int(self.get_parameter('v_min').value)
            ],
            dtype=np.uint8
        )

        upper = np.array(
            [
                int(self.get_parameter('h_max').value),
                int(self.get_parameter('s_max').value),
                int(self.get_parameter('v_max').value)
            ],
            dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower, upper)

        # Remove very low-saturation reflections that may accidentally
        # enter the hue interval.
        _, sat_mask = cv2.threshold(
            hsv[:, :, 1],
            int(self.get_parameter('s_min').value),
            255,
            cv2.THRESH_BINARY
        )
        mask = cv2.bitwise_and(mask, sat_mask)

        # Initial light cleanup.
        mask = cv2.medianBlur(mask, 3)

        # First contour estimate to adapt morphology to object size.
        initial_contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not initial_contours:
            return None, mask

        largest_initial = max(
            initial_contours,
            key=cv2.contourArea
        )
        _, _, bw0, bh0 = cv2.boundingRect(largest_initial)
        characteristic = max(5, min(bw0, bh0))

        close_ratio = float(
            self.get_parameter('dice_close_ratio').value
        )
        open_ratio = float(
            self.get_parameter('dice_open_ratio').value
        )

        close_size = self.make_odd(
            max(3, int(round(characteristic * close_ratio)))
        )
        open_size = self.make_odd(
            max(3, int(round(characteristic * open_ratio)))
        )

        close_size = min(close_size, 11)
        open_size = min(open_size, 7)

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size)
        )
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_size, open_size)
        )

        # Closing first is important: black pips create holes/notches in
        # the yellow region, so we fill them before removing isolated noise.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = float(
            self.get_parameter('min_dice_area').value
        )
        min_ratio = float(
            self.get_parameter('min_aspect_ratio').value
        )
        max_ratio = float(
            self.get_parameter('max_aspect_ratio').value
        )

        valid = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]

            if rw <= 0 or rh <= 0:
                continue

            ratio = max(rw, rh) / min(rw, rh)

            # Convert user parameters to an equivalent symmetric criterion.
            # min_aspect_ratio/max_aspect_ratio are retained for compatibility.
            if ratio > max(max_ratio, 1.0 / max(min_ratio, 1e-6)):
                continue

            valid.append(contour)

        if not valid:
            return None, mask

        contour = max(valid, key=cv2.contourArea)

        # Use convex hull for the complete die silhouette. This avoids black
        # pips cutting notches into the object boundary.
        contour = cv2.convexHull(contour)

        M = cv2.moments(contour)
        if M['m00'] == 0:
            return None, mask

        u = int(round(M['m10'] / M['m00']))
        v = int(round(M['m01'] / M['m00']))

        # Always rectangular internal bounding box.
        rect = cv2.minAreaRect(contour)

        detection = {
            'contour': contour,
            'rect': rect,
            'u': u,
            'v': v
        }

        return detection, mask

    # ================================================================
    # RAW PIP DETECTION ON COMPLETE DICE
    # ================================================================
    def detect_raw_pips(self, frame, dice_contour, dice_rect):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        dice_mask = np.zeros(
            gray.shape,
            dtype=np.uint8
        )
        cv2.drawContours(
            dice_mask,
            [dice_contour],
            -1,
            255,
            cv2.FILLED
        )

        threshold_value = int(
            self.get_parameter('raw_pip_gray_threshold').value
        )

        _, dark = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV
        )

        dark = cv2.bitwise_and(dark, dice_mask)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            dark,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        rw, rh = dice_rect[1]
        dice_box_area = max(rw * rh, 1.0)

        min_area = dice_box_area * float(
            self.get_parameter('raw_pip_min_area_ratio').value
        )
        max_area = dice_box_area * float(
            self.get_parameter('raw_pip_max_area_ratio').value
        )
        min_circularity = float(
            self.get_parameter('raw_pip_min_circularity').value
        )
        max_aspect = float(
            self.get_parameter('raw_pip_max_aspect_ratio').value
        )

        candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = (
                4.0 * np.pi * area /
                (perimeter * perimeter)
            )

            if circularity < min_circularity:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw <= 0 or bh <= 0:
                continue

            aspect = max(bw, bh) / float(min(bw, bh))
            if aspect > max_aspect:
                continue

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue

            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])

            candidates.append({
                'center': np.array([cx, cy], dtype=np.float32),
                'area': float(area),
                'contour': cnt,
                'circularity': float(circularity)
            })

        return candidates, dark

    # ================================================================
    # SELECT TOP-FACE PIPS
    # ================================================================
    def select_top_pips(
        self,
        candidates,
        dice_rect,
        dice_center,
        image_shape
    ):
        if not candidates:
            return []

        h, w = image_shape[:2]
        dice_center = np.array(
            dice_center,
            dtype=np.float32
        )

        if self.processing_camera_matrix is not None:
            principal = np.array(
                [
                    self.processing_camera_matrix[0, 2],
                    self.processing_camera_matrix[1, 2]
                ],
                dtype=np.float32
            )
        else:
            principal = np.array(
                [w / 2.0, h / 2.0],
                dtype=np.float32
            )

        outward = dice_center - principal
        outward_norm = float(np.linalg.norm(outward))

        # Near the optical axis there is very little visible side face;
        # therefore all valid dark blobs are considered top candidates.
        diag = float(np.hypot(w, h))
        normalized_radial_distance = (
            outward_norm / max(0.5 * diag, 1.0)
        )

        if outward_norm < 1e-6 or normalized_radial_distance < 0.08:
            selected = list(candidates)
        else:
            outward_unit = outward / outward_norm

            projections = np.array(
                [
                    float(np.dot(c['center'] - dice_center, outward_unit))
                    for c in candidates
                ],
                dtype=np.float32
            )

            rw, rh = dice_rect[1]
            characteristic = max(min(rw, rh), 1.0)
            gate = float(
                self.get_parameter('top_pip_radial_gate').value
            ) * characteristic

            # Side pips are expected mainly on the inward side of the die,
            # i.e. with strongly negative radial projection.
            selected = [
                c for c, projection in zip(candidates, projections)
                if projection >= -gate
            ]

            if not selected:
                selected = list(candidates)

        # Robust area-consistency filter. Top pips on one face should have
        # reasonably similar apparent sizes.
        if len(selected) >= 2:
            areas = np.array(
                [c['area'] for c in selected],
                dtype=np.float32
            )
            median_area = float(np.median(areas))

            coherent = [
                c for c in selected
                if 0.35 <= c['area'] / max(median_area, 1e-6) <= 2.80
            ]

            if coherent:
                selected = coherent

        # A physical die cannot show more than 6 pips on one face.
        if len(selected) > 6:
            median_area = float(np.median(
                [c['area'] for c in selected]
            ))

            def score(c):
                area_error = abs(
                    np.log(max(c['area'] / median_area, 1e-6))
                )
                shape_error = 1.0 - min(c['circularity'], 1.0)
                return area_error + 0.4 * shape_error

            selected = sorted(selected, key=score)[:6]

        return selected

    # ================================================================
    # ELLIPSE + RECTANGULAR TOP-FACE BOX
    # ================================================================
    def estimate_top_face_from_pips(
        self,
        frame,
        raw_candidates,
        dice_rect,
        dice_center
    ):
        selected = self.select_top_pips(
            raw_candidates,
            dice_rect,
            dice_center,
            frame.shape
        )

        if not selected:
            return self.fallback_top_face_box(
                frame,
                dice_rect,
                dice_center
            )

        # Collect contour boundary points rather than only centroids. This
        # lets fitEllipse work even when the face contains only 1-4 pips.
        all_points = []
        for c in selected:
            pts = c['contour'].reshape(-1, 2)
            if len(pts) > 0:
                all_points.append(pts)

        if not all_points:
            return self.fallback_top_face_box(
                frame,
                dice_rect,
                dice_center
            )

        points = np.vstack(all_points).astype(np.float32)

        rw, rh = dice_rect[1]
        dice_long = max(rw, rh)
        dice_short = min(rw, rh)

        # ------------------------------------------------------------
        # Fit ellipse to pip group.
        # ------------------------------------------------------------
        if len(points) >= 5:
            ellipse = cv2.fitEllipse(
                points.reshape(-1, 1, 2)
            )
            e_center, e_size, e_angle = ellipse
        else:
            # This should be rare because contour boundaries normally
            # provide >5 points, but keep a deterministic fallback.
            pip_centres = np.array(
                [c['center'] for c in selected],
                dtype=np.float32
            )
            e_center = tuple(np.mean(pip_centres, axis=0))
            e_size = (
                max(0.5 * dice_short, 1.0),
                max(0.5 * dice_short, 1.0)
            )
            e_angle = float(dice_rect[2])

        e_center = np.array(e_center, dtype=np.float32)
        a = max(float(e_size[0]) * 0.5, 1.0)
        b = max(float(e_size[1]) * 0.5, 1.0)

        # ------------------------------------------------------------
        # Make the fitted ellipse actually contain ALL selected pip
        # contour points. OpenCV fitEllipse is least-squares and does not
        # guarantee containment, so scale its axes when required.
        # ------------------------------------------------------------
        theta = np.deg2rad(float(e_angle))
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array(
            [[c, s], [-s, c]],
            dtype=np.float32
        )

        local = (points - e_center) @ R.T
        normalized = np.sqrt(
            (local[:, 0] / a) ** 2
            + (local[:, 1] / b) ** 2
        )

        containment_scale = max(
            1.0,
            float(np.max(normalized))
        )
        a *= containment_scale
        b *= containment_scale

        # ------------------------------------------------------------
        # Perspective prior:
        # centre image -> circle-like top face
        # image borders -> increasingly elliptical top face
        # ------------------------------------------------------------
        h, w = frame.shape[:2]

        if self.processing_camera_matrix is not None:
            principal = np.array(
                [
                    self.processing_camera_matrix[0, 2],
                    self.processing_camera_matrix[1, 2]
                ],
                dtype=np.float32
            )
        else:
            principal = np.array(
                [w / 2.0, h / 2.0],
                dtype=np.float32
            )

        radial = float(
            np.linalg.norm(
                np.array(dice_center, dtype=np.float32) - principal
            )
        )
        radial_max = max(0.5 * float(np.hypot(w, h)), 1.0)
        radial_norm = float(np.clip(radial / radial_max, 0.0, 1.0))

        perspective_strength = float(
            self.get_parameter('ellipse_perspective_strength').value
        )

        expected_minor_major = np.clip(
            1.0 - perspective_strength * radial_norm,
            0.60,
            1.0
        )

        major = max(a, b)
        minor = min(a, b)

        # Do not force the fitted ellipse to become more eccentric than
        # observed; only prevent an implausibly thin top-face estimate.
        minimum_minor = major * expected_minor_major
        minor = max(minor, minimum_minor)

        if a >= b:
            a = major
            b = minor
        else:
            b = major
            a = minor

        # ------------------------------------------------------------
        # Convert ellipse to a rectangular oriented top-face box.
        # Start from pip ellipse, enlarge, then enforce plausible dice size.
        # ------------------------------------------------------------
        box_scale = float(
            self.get_parameter('top_face_box_scale').value
        )
        min_size_ratio = float(
            self.get_parameter('top_face_min_size_ratio').value
        )
        max_size_ratio = float(
            self.get_parameter('top_face_max_size_ratio').value
        )

        ellipse_width = 2.0 * a * box_scale
        ellipse_height = 2.0 * b * box_scale

        min_face_size = min_size_ratio * dice_short
        max_face_size = max_size_ratio * dice_long

        face_w = float(np.clip(
            ellipse_width,
            min_face_size,
            max_face_size
        ))
        face_h = float(np.clip(
            ellipse_height,
            min_face_size,
            max_face_size
        ))

        # With 1-2 pips the fitted ellipse centre follows the pips rather
        # than the physical face centre. Blend it toward a geometric prior.
        geometric_center = self.estimate_top_face_center(
            frame,
            dice_rect,
            dice_center
        )

        n_pips = len(selected)
        if n_pips <= 2:
            alpha = 0.35
        elif n_pips == 3:
            alpha = 0.65
        else:
            alpha = 0.90

        face_center = (
            alpha * e_center
            + (1.0 - alpha) * geometric_center
        )

        face_rect = (
            (float(face_center[0]), float(face_center[1])),
            (face_w, face_h),
            float(e_angle)
        )

        box = cv2.boxPoints(face_rect).astype(np.float32)
        box = self.order_points(box)

        ellipse_out = (
            (float(e_center[0]), float(e_center[1])),
            (2.0 * a, 2.0 * b),
            float(e_angle)
        )

        return box, ellipse_out

    # ================================================================
    # TOP-FACE GEOMETRIC PRIOR
    # ================================================================
    def estimate_top_face_center(self, frame, dice_rect, dice_center):
        h, w = frame.shape[:2]
        dice_center = np.array(dice_center, dtype=np.float32)

        if self.processing_camera_matrix is not None:
            principal = np.array(
                [
                    self.processing_camera_matrix[0, 2],
                    self.processing_camera_matrix[1, 2]
                ],
                dtype=np.float32
            )
        else:
            principal = np.array(
                [w / 2.0, h / 2.0],
                dtype=np.float32
            )

        outward = dice_center - principal
        norm = float(np.linalg.norm(outward))

        if norm < 1e-6:
            return dice_center

        outward_unit = outward / norm
        rw, rh = dice_rect[1]
        characteristic = max(min(rw, rh), 1.0)

        # Top face is displaced slightly outward relative to the complete
        # die silhouette when lateral faces are visible.
        shift = 0.10 * characteristic

        return dice_center + shift * outward_unit

    # ================================================================
    # FALLBACK TOP-FACE RECTANGLE
    # ================================================================
    def fallback_top_face_box(self, frame, dice_rect, dice_center):
        center = self.estimate_top_face_center(
            frame,
            dice_rect,
            dice_center
        )

        rw, rh = dice_rect[1]
        short = max(min(rw, rh), 1.0)
        size_ratio = float(
            self.get_parameter('top_face_min_size_ratio').value
        )
        face_size = short * size_ratio

        rect = (
            (float(center[0]), float(center[1])),
            (face_size, face_size),
            float(dice_rect[2])
        )

        box = cv2.boxPoints(rect).astype(np.float32)
        return self.order_points(box), None

    # ================================================================
    # ORDER FOUR CORNERS: TL, TR, BR, BL
    # ================================================================
    def order_points(self, pts):
        pts = np.asarray(
            pts,
            dtype=np.float32
        ).reshape(4, 2)

        center = np.mean(pts, axis=0)
        angles = np.arctan2(
            pts[:, 1] - center[1],
            pts[:, 0] - center[0]
        )
        ordered = pts[np.argsort(angles)]

        # Rotate sequence so first point is visually top-left.
        start = np.argmin(
            ordered[:, 0] + ordered[:, 1]
        )
        ordered = np.roll(ordered, -start, axis=0)

        # Ensure clockwise TL, TR, BR, BL ordering.
        if ordered[1, 0] < ordered[-1, 0]:
            ordered = np.array(
                [
                    ordered[0],
                    ordered[-1],
                    ordered[-2],
                    ordered[-3]
                ],
                dtype=np.float32
            )

        return ordered.astype(np.float32)

    # ================================================================
    # RECTIFY TOP FACE
    # ================================================================
    def rectify_dice_face(self, frame, top_face_box):
        if top_face_box is None:
            return None, None

        src = self.order_points(top_face_box)
        size = self.rectified_size

        dst = np.array(
            [
                [0.0, 0.0],
                [size - 1.0, 0.0],
                [size - 1.0, size - 1.0],
                [0.0, size - 1.0]
            ],
            dtype=np.float32
        )

        H = cv2.getPerspectiveTransform(src, dst)

        rectified = cv2.warpPerspective(
            frame,
            H,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rectified, H

    # ================================================================
    # FINAL PIP COUNT ON RECTIFIED TOP FACE
    # ================================================================

    def count_pips(self, rectified):

        # ============================================================
        # PREPROCESSING
        # ============================================================

        gray = cv2.cvtColor(
            rectified,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        hsv = cv2.cvtColor(
            rectified,
            cv2.COLOR_BGR2HSV
        )

        h_img, w_img = gray.shape

        image_area = float(
            h_img * w_img
        )

        # ============================================================
        # COMMON PARAMETERS
        # ============================================================

        min_area = (
            image_area
            *
            float(
                self.get_parameter(
                    'pip_min_area_ratio'
                ).value
            )
        )

        max_area = (
            image_area
            *
            float(
                self.get_parameter(
                    'pip_max_area_ratio'
                ).value
            )
        )

        min_circularity = float(
            self.get_parameter(
                'pip_min_circularity'
            ).value
        )

        # ============================================================
        # HELPER FUNCTION:
        # EXTRACT VALID BLOBS FROM A BINARY MASK
        # ============================================================

        def extract_candidates(
            binary_mask,
            mode
        ):

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []

            for cnt in contours:

                # ----------------------------------------------------
                # AREA
                # ----------------------------------------------------

                area = cv2.contourArea(
                    cnt
                )

                if (
                    area < min_area
                    or
                    area > max_area
                ):
                    continue

                # ----------------------------------------------------
                # CIRCULARITY
                # ----------------------------------------------------

                perimeter = cv2.arcLength(
                    cnt,
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

                # ----------------------------------------------------
                # ASPECT RATIO
                # ----------------------------------------------------

                x, y, bw, bh = cv2.boundingRect(
                    cnt
                )

                if (
                    bw <= 0
                    or
                    bh <= 0
                ):
                    continue

                aspect = (
                    max(
                        bw,
                        bh
                    )
                    /
                    float(
                        min(
                            bw,
                            bh
                        )
                    )
                )

                # Use your tuned value here
                if aspect > 1.60:
                    continue

                # ----------------------------------------------------
                # SOLIDITY
                # ----------------------------------------------------

                hull = cv2.convexHull(
                    cnt
                )

                hull_area = cv2.contourArea(
                    hull
                )

                if hull_area <= 0:
                    continue

                solidity = (
                    area
                    /
                    hull_area
                )

                if solidity < 0.80:
                    continue

                # ----------------------------------------------------
                # CENTER
                # ----------------------------------------------------

                M = cv2.moments(
                    cnt
                )

                if M['m00'] == 0:
                    continue

                cx = int(
                    round(
                        M['m10']
                        /
                        M['m00']
                    )
                )

                cy = int(
                    round(
                        M['m01']
                        /
                        M['m00']
                    )
                )

                # ----------------------------------------------------
                # MEAN INTENSITY
                # ----------------------------------------------------

                single_mask = np.zeros_like(
                    gray
                )

                cv2.drawContours(
                    single_mask,
                    [cnt],
                    -1,
                    255,
                    cv2.FILLED
                )

                mean_intensity = cv2.mean(
                    gray,
                    mask=single_mask
                )[0]

                candidates.append(
                    {
                        'center': (
                            cx,
                            cy
                        ),
                        'area': float(
                            area
                        ),
                        'circularity': float(
                            circularity
                        ),
                        'solidity': float(
                            solidity
                        ),
                        'mean_intensity': float(
                            mean_intensity
                        ),
                        'mode': mode
                    }
                )

            return candidates

        # ============================================================
        # 1. BLACK PIP DETECTION
        # ============================================================

        threshold_value = int(
            self.get_parameter(
                'pip_gray_threshold'
            ).value
        )

        _, black_mask = cv2.threshold(
            gray,
            threshold_value,
            255,
            cv2.THRESH_BINARY_INV
        )

        # ============================================================
        # REMOVE IMAGE BORDER
        # ============================================================

        margin = int(
            round(
                float(
                    self.get_parameter(
                        'pip_border_margin_ratio'
                    ).value
                )
                *
                min(
                    h_img,
                    w_img
                )
            )
        )

        if margin > 0:

            black_mask[
                :margin,
                :
            ] = 0

            black_mask[
                -margin:,
                :
            ] = 0

            black_mask[
                :,
                :margin
            ] = 0

            black_mask[
                :,
                -margin:
            ] = 0

        # ============================================================
        # MORPHOLOGY
        # ============================================================

        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_OPEN,
            kernel_open,
            iterations=1
        )

        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            kernel_close,
            iterations=1
        )

        black_candidates = extract_candidates(
            black_mask,
            'black'
        )

        # ============================================================
        # IF BLACK PIPS WERE FOUND -> USE THEM
        # ============================================================

        if len(black_candidates) > 0:

            candidates = black_candidates

            pip_mask = black_mask

            self.get_logger().info(
                'Pip detection mode: BLACK'
            )

        # ============================================================
        # 2. NO BLACK PIPS -> SEARCH FOR WHITE PIPS
        # ============================================================

        else:

            self.get_logger().info(
                'No black pips detected. '
                'Trying WHITE/specular pip detection.'
            )

            S = hsv[
                :,
                :,
                1
            ]

            V = hsv[
                :,
                :,
                2
            ]

            white_v_min = int(
                self.get_parameter(
                    'white_pip_v_min'
                ).value
            )

            white_s_max = int(
                self.get_parameter(
                    'white_pip_s_max'
                ).value
            )

            # ========================================================
            # WHITE CONDITION
            #
            # Very bright + low saturation
            # ========================================================

            white_condition = (
                (V >= white_v_min)
                &
                (S <= white_s_max)
            )

            white_mask = (
                white_condition.astype(
                    np.uint8
                )
                *
                255
            )

            # ========================================================
            # REMOVE BORDER
            # ========================================================

            if margin > 0:

                white_mask[
                    :margin,
                    :
                ] = 0

                white_mask[
                    -margin:,
                    :
                ] = 0

                white_mask[
                    :,
                    :margin
                ] = 0

                white_mask[
                    :,
                    -margin:
                ] = 0

            # ========================================================
            # MORPHOLOGY
            # ========================================================

            white_mask = cv2.morphologyEx(
                white_mask,
                cv2.MORPH_OPEN,
                kernel_open,
                iterations=1
            )

            white_mask = cv2.morphologyEx(
                white_mask,
                cv2.MORPH_CLOSE,
                kernel_close,
                iterations=1
            )

            white_candidates = extract_candidates(
                white_mask,
                'white'
            )

            candidates = white_candidates

            pip_mask = white_mask

            if len(white_candidates) > 0:

                self.get_logger().info(
                    f'Pip detection mode: WHITE - '
                    f'{len(white_candidates)} candidates'
                )

            else:

                self.get_logger().warn(
                    'No black or white pips detected.'
                )

                return (
                    0,
                    [],
                    pip_mask
                )

        # ============================================================
        # AREA CONSISTENCY
        # ============================================================

        areas = np.array(
            [
                c['area']
                for c in candidates
            ],
            dtype=np.float32
        )

        median_area = float(
            np.median(
                areas
            )
        )

        filtered = [
            c
            for c in candidates
            if (
                0.50
                <=
                c['area']
                /
                max(
                    median_area,
                    1e-6
                )
                <=
                1.80
            )
        ]

        if not filtered:

            filtered = candidates

        # ============================================================
        # MAXIMUM 6 PIPS
        # ============================================================

        if len(filtered) > 6:

            mode = filtered[0]['mode']

            def score(candidate):

                area_error = abs(
                    np.log(
                        max(
                            candidate['area']
                            /
                            median_area,
                            1e-6
                        )
                    )
                )

                circularity_error = (
                    1.0
                    -
                    min(
                        candidate[
                            'circularity'
                        ],
                        1.0
                    )
                )

                solidity_error = (
                    1.0
                    -
                    candidate[
                        'solidity'
                    ]
                )

                # -----------------------------------------------
                # Intensity criterion depends on detection mode
                # -----------------------------------------------

                if mode == 'black':

                    # Lower intensity is better
                    intensity_error = (
                        candidate[
                            'mean_intensity'
                        ]
                        /
                        255.0
                    )

                else:

                    # Higher intensity is better
                    intensity_error = (
                        1.0
                        -
                        candidate[
                            'mean_intensity'
                        ]
                        /
                        255.0
                    )

                return (
                    1.4
                    *
                    area_error
                    +
                    0.8
                    *
                    circularity_error
                    +
                    0.8
                    *
                    solidity_error
                    +
                    0.8
                    *
                    intensity_error
                )

            filtered = sorted(
                filtered,
                key=score
            )[:6]

        # ============================================================
        # CENTERS
        # ============================================================

        pip_centers = [
            candidate[
                'center'
            ]
            for candidate in filtered
        ]

        pip_centers.sort(
            key=lambda p: (
                p[1],
                p[0]
            )
        )

        return (
            len(
                pip_centers
            ),
            pip_centers,
            pip_mask
        )
    
    # ================================================================
    # DRAW ORIENTED DICE BOUNDING BOX + LOCAL X/Y AXES
    # ================================================================
    def draw_dice_oriented_box(self, image, dice_rect):
        """Draw the complete-dice oriented rectangle and its local frame.

        The local x axis is aligned with the longest side of the oriented
        rectangle returned by cv2.minAreaRect(). The local y axis is
        perpendicular to x. This function changes only visualization; it
        does not affect detection, rectification, pose estimation or pip
        counting.
        """

        if dice_rect is None:
            return

        center, size, angle = dice_rect
        cx = float(center[0])
        cy = float(center[1])
        width = float(size[0])
        height = float(size[1])

        if width <= 0.0 or height <= 0.0:
            return

        # ------------------------------------------------------------
        # Oriented bounding box of the complete dice
        # ------------------------------------------------------------
        box = cv2.boxPoints(dice_rect)
        box_int = np.round(box).astype(np.int32)

        cv2.polylines(
            image,
            [box_int],
            True,
            (0, 255, 255),
            3
        )

        # ------------------------------------------------------------
        # Local frame
        #
        # OpenCV's angle convention depends on which side is reported
        # as width/height. We always define x along the longest side.
        # ------------------------------------------------------------
        local_angle = float(angle)

        if width < height:
            local_angle += 90.0
            x_length = height
            y_length = width
        else:
            x_length = width
            y_length = height

        theta_x = np.deg2rad(local_angle)
        theta_y = theta_x + np.pi / 2.0

        x_axis_length = 0.42 * x_length
        y_axis_length = 0.42 * y_length

        origin = (
            int(round(cx)),
            int(round(cy))
        )

        x_end = (
            int(round(cx + x_axis_length * np.cos(theta_x))),
            int(round(cy + x_axis_length * np.sin(theta_x)))
        )

        y_end = (
            int(round(cx + y_axis_length * np.cos(theta_y))),
            int(round(cy + y_axis_length * np.sin(theta_y)))
        )

        # x axis: red
        cv2.arrowedLine(
            image,
            origin,
            x_end,
            (0, 0, 255),
            3,
            tipLength=0.20
        )

        # y axis: blue
        cv2.arrowedLine(
            image,
            origin,
            y_end,
            (255, 0, 0),
            3,
            tipLength=0.20
        )

        cv2.putText(
            image,
            'x',
            (x_end[0] + 5, x_end[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            image,
            'y',
            (y_end[0] + 5, y_end[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # Local-frame origin
        cv2.circle(
            image,
            origin,
            4,
            (255, 255, 255),
            -1
        )

    # ================================================================
    # PIXEL -> CAMERA POSITION
    # ================================================================
    def pixel_to_camera_position(self, u, v):
        if self.processing_camera_matrix is None:
            K = self.camera_matrix
        else:
            K = self.processing_camera_matrix

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        if fx == 0 or fy == 0:
            return None

        x_n = (float(u) - cx) / fx
        y_n = (float(v) - cy) / fy

        Z = self.dice_z
        X = x_n * Z
        Y = y_n * Z

        return X, Y, Z

    # ================================================================
    # TF / FULL POSE ESTIMATION
    # ================================================================
    def estimate_dice_poses(
        self,
        X,
        Y,
        Z,
        dice_rect,
        stamp
    ):
        """Estimate die poses in camera_link and world.

        No additional optical-frame transform is applied in this version.
        The X, Y, Z coordinates produced by the pinhole model are interpreted
        directly in camera_link_frame. The robot TF tree only needs to provide
        world <- camera_link.

        The die is assumed to rest flat on the table, therefore its world z axis
        is aligned with world +Z and only yaw is estimated from the oriented box.
        """

        if self.processing_camera_matrix is None and self.camera_matrix is None:
            self.get_logger().warn(
                'Camera intrinsics are not available; cannot estimate full pose.'
            )
            return None, None, None

        # ------------------------------------------------------------
        # Lookup world <- camera_link
        # ------------------------------------------------------------
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_link_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)
            )
        except Exception as exc:
            self.get_logger().warn(
                f'Could not get TF {self.world_frame} <- '
                f'{self.camera_link_frame}: {exc}'
            )
            return None, None, None

        R_world_link, t_world_link = self.transform_to_matrix(transform)

        # ------------------------------------------------------------
        # Position: camera_link -> world
        # ------------------------------------------------------------
        p_link = np.array(
            [float(X), float(Y), float(Z)],
            dtype=np.float64
        )

        p_world = R_world_link @ p_link + t_world_link

        # ------------------------------------------------------------
        # Image direction of the die local x axis. Keep exactly the
        # same convention used for visualization.
        # ------------------------------------------------------------
        center, size, angle = dice_rect
        cx = float(center[0])
        cy = float(center[1])
        width = float(size[0])
        height = float(size[1])

        if width <= 0.0 or height <= 0.0:
            return None, None, None

        local_angle = float(angle)

        if width < height:
            local_angle += 90.0
            axis_length = 0.42 * height
        else:
            axis_length = 0.42 * width

        theta = np.deg2rad(local_angle)

        axis_pixel = np.array(
            [
                cx + axis_length * np.cos(theta),
                cy + axis_length * np.sin(theta)
            ],
            dtype=np.float64
        )

        if self.processing_camera_matrix is None:
            K = self.camera_matrix
        else:
            K = self.processing_camera_matrix

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cxi = float(K[0, 2])
        cyi = float(K[1, 2])

        if fx == 0.0 or fy == 0.0:
            return None, None, None

        # Direction associated with the chosen image-axis endpoint.
        # It is interpreted directly in camera_link_frame.
        ray_link = np.array(
            [
                (axis_pixel[0] - cxi) / fx,
                (axis_pixel[1] - cyi) / fy,
                1.0
            ],
            dtype=np.float64
        )
        ray_link /= max(np.linalg.norm(ray_link), 1e-12)

        # Direction vectors are rotated, never translated.
        ray_world = R_world_link @ ray_link
        camera_origin_world = t_world_link

        # ------------------------------------------------------------
        # Intersect the ray with the horizontal plane through the die.
        # The die is assumed flat on the table, so only world yaw is free.
        # ------------------------------------------------------------
        if abs(ray_world[2]) > 1e-9:
            lam = (p_world[2] - camera_origin_world[2]) / ray_world[2]
            axis_point_world = camera_origin_world + lam * ray_world
            x_world = axis_point_world - p_world
        else:
            image_direction_link = np.array(
                [
                    np.cos(theta) / fx,
                    np.sin(theta) / fy,
                    0.0
                ],
                dtype=np.float64
            )
            x_world = R_world_link @ image_direction_link

        # The die lies on the table: keep only the world XY direction.
        x_world[2] = 0.0
        x_world_norm = float(np.linalg.norm(x_world[:2]))

        if x_world_norm < 1e-9:
            self.get_logger().warn(
                'Could not infer die yaw from the oriented bounding box.'
            )
            return None, None, None

        x_world /= x_world_norm
        yaw_world = float(np.arctan2(x_world[1], x_world[0]))

        # ------------------------------------------------------------
        # World orientation: roll = pitch = 0, yaw estimated above.
        # ------------------------------------------------------------
        cyaw = np.cos(0.5 * yaw_world)
        syaw = np.sin(0.5 * yaw_world)
        q_world_dice = np.array(
            [0.0, 0.0, syaw, cyaw],
            dtype=np.float64
        )

        pose_world = PoseStamped()
        pose_world.header.stamp = stamp
        pose_world.header.frame_id = self.world_frame
        pose_world.pose.position.x = float(p_world[0])
        pose_world.pose.position.y = float(p_world[1])
        pose_world.pose.position.z = float(p_world[2])
        pose_world.pose.orientation.x = float(q_world_dice[0])
        pose_world.pose.orientation.y = float(q_world_dice[1])
        pose_world.pose.orientation.z = float(q_world_dice[2])
        pose_world.pose.orientation.w = float(q_world_dice[3])

        # ------------------------------------------------------------
        # Pose of the same die frame expressed in camera_link.
        # ------------------------------------------------------------
        c_yaw = np.cos(yaw_world)
        s_yaw = np.sin(yaw_world)
        R_world_dice = np.array(
            [
                [c_yaw, -s_yaw, 0.0],
                [s_yaw,  c_yaw, 0.0],
                [0.0,    0.0,   1.0]
            ],
            dtype=np.float64
        )

        R_link_dice = R_world_link.T @ R_world_dice
        q_link_dice = self.rotation_matrix_to_quaternion(R_link_dice)

        pose_camera = PoseStamped()
        pose_camera.header.stamp = stamp
        pose_camera.header.frame_id = self.camera_link_frame
        pose_camera.pose.position.x = float(p_link[0])
        pose_camera.pose.position.y = float(p_link[1])
        pose_camera.pose.position.z = float(p_link[2])
        pose_camera.pose.orientation.x = float(q_link_dice[0])
        pose_camera.pose.orientation.y = float(q_link_dice[1])
        pose_camera.pose.orientation.z = float(q_link_dice[2])
        pose_camera.pose.orientation.w = float(q_link_dice[3])

        return pose_camera, pose_world, yaw_world

    # ================================================================
    # TRANSFORM -> ROTATION MATRIX + TRANSLATION
    # ================================================================
    @staticmethod
    def transform_to_matrix(transform):
        q = transform.transform.rotation
        x = float(q.x)
        y = float(q.y)
        z = float(q.z)
        w = float(q.w)

        norm = np.sqrt(x*x + y*y + z*z + w*w)
        if norm < 1e-12:
            x, y, z, w = 0.0, 0.0, 0.0, 1.0
        else:
            x /= norm
            y /= norm
            z /= norm
            w /= norm

        R = np.array(
            [
                [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
                [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
                [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
            ],
            dtype=np.float64
        )

        t = transform.transform.translation
        translation = np.array(
            [float(t.x), float(t.y), float(t.z)],
            dtype=np.float64
        )

        return R, translation

    # ================================================================
    # ROTATION MATRIX -> QUATERNION [x, y, z, w]
    # ================================================================
    @staticmethod
    def rotation_matrix_to_quaternion(R):
        R = np.asarray(R, dtype=np.float64)
        trace = float(np.trace(R))

        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s

        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s

        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s

        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        q /= max(np.linalg.norm(q), 1e-12)
        return q

    # ================================================================
    # BROADCAST DICE TF FOR RVIZ
    # ================================================================
    def broadcast_dice_tf(self, pose_world):
        """Broadcast world -> dice_frame from the estimated world pose."""

        if pose_world is None:
            return

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.world_frame
        transform.child_frame_id = self.dice_frame

        transform.transform.translation.x = float(
            pose_world.pose.position.x
        )
        transform.transform.translation.y = float(
            pose_world.pose.position.y
        )
        transform.transform.translation.z = float(
            pose_world.pose.position.z
        )

        transform.transform.rotation.x = float(
            pose_world.pose.orientation.x
        )
        transform.transform.rotation.y = float(
            pose_world.pose.orientation.y
        )
        transform.transform.rotation.z = float(
            pose_world.pose.orientation.z
        )
        transform.transform.rotation.w = float(
            pose_world.pose.orientation.w
        )

        self.last_dice_transform = transform
        self.dice_tf_broadcaster.sendTransform(
            transform
        )

    # ================================================================
    # PERIODIC TF REPUBLISH FOR RVIZ
    # ================================================================
    def republish_dice_tf(self):
        """Keep the one-shot detected dice TF alive for RViz."""

        if self.last_dice_transform is None:
            return

        self.last_dice_transform.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.dice_tf_broadcaster.sendTransform(
            self.last_dice_transform
        )

    # ================================================================
    # PUBLISH POSITION
    # ================================================================
    def publish_position(self, X, Y, Z, stamp):
        """Publish the die position expressed directly in camera_link.

        No additional optical-frame rotation is applied.
        """
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_link_frame
        msg.point.x = float(X)
        msg.point.y = float(Y)
        msg.point.z = float(Z)
        self.position_pub.publish(msg)

    # ================================================================
    # UTILITY
    # ================================================================
    @staticmethod
    def make_odd(value):
        value = int(value)
        if value % 2 == 0:
            value += 1
        return value


    # ================================================================
    # CLEANUP
    # ================================================================
    def destroy_node(self):
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None
        super().destroy_node()


# ====================================================================
# MAIN
# ====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = DiceDetectorService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
