#!/usr/bin/env python3
"""Automatic hand-eye + depth calibration node.

Drives the arm through a configured set of poses, detects colored markers
in the camera image, looks up their ground-truth 3D position via TF, runs
Depth-Anything-V2 for a relative depth sample per marker pixel, and then
solves:

  1. Hand-Eye (T_cam_base) via cv2.solvePnP SQPNP
  2. Depth parameters (a, b) via Z_cam = a / d_rel + b fit

Services
--------
~/run_sequence     (std_srvs/Trigger)  drive arm + collect + solve
~/recalibrate      (std_srvs/Trigger)  solve from currently-buffered points
~/reset            (std_srvs/Trigger)  drop all buffered points
~/save             (std_srvs/Trigger)  write results to the canonical YAMLs
~/load             (std_srvs/Trigger)  read current values from the YAMLs

Topics
------
/ba_calib/status              std_msgs/String
/ba_calib/datapoints          std_msgs/Int32
/ba_calib/reprojection_px     std_msgs/Float32
/ba_calib/depth_rmse_m        std_msgs/Float32
/ba_calib/hand_eye            geometry_msgs/TransformStamped  (latched)
/ba_calib/depth_params        std_msgs/Float32MultiArray      (latched)
/ba_calib/debug/image         sensor_msgs/Image
"""

import os
import threading
import time
import traceback
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from std_msgs.msg import Float32, Float32MultiArray, Int32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, LookupException, TransformListener

from .data_collector import DataCollector, StillnessDetector
from .marker_detector import MarkerConfig, MarkerDetector
from .persistence import write_depth, write_hand_eye
from .sequence_runner import CalibPose, SequenceRunner
from .solvers import (DataPoint, HandEyeResult, invert_transform,
                      solve_depth, solve_hand_eye)


class AutoCalibNode(Node):
    def __init__(self):
        super().__init__('ba_autocalib_node')
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter('image_topic',
                               '/ba_overview_camera/image_raw/compressed')
        self.declare_parameter('camera_info_topic',
                               '/ba_overview_camera/camera_info')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rectify', True)
        self.declare_parameter('marker_config_file', '')
        self.declare_parameter('calib_poses_file', '')
        self.declare_parameter('perception_yaml_path', '')
        self.declare_parameter('depth_yaml_path', '')
        self.declare_parameter('backup_dir',
                               os.path.expanduser(
                                   '~/.ros/ba_calibration/backup'))
        self.declare_parameter('planning_group', 'arm')
        self.declare_parameter('velocity_scaling', 0.2)
        self.declare_parameter('acceleration_scaling', 0.2)
        self.declare_parameter('settle_s', 0.7)
        self.declare_parameter('depth_model_id',
                               'depth-anything/Depth-Anything-V2-Small-hf')
        self.declare_parameter('depth_device', 'cpu')
        self.declare_parameter('min_points_to_solve', 8)

        self._image_topic = self.get_parameter('image_topic').value
        self._info_topic = self.get_parameter('camera_info_topic').value
        self._base_frame = self.get_parameter('base_frame').value
        self._rectify = self.get_parameter('rectify').value
        self._marker_cfg_path = self.get_parameter('marker_config_file').value
        self._poses_path = self.get_parameter('calib_poses_file').value
        self._perception_yaml = self.get_parameter('perception_yaml_path').value
        self._depth_yaml = self.get_parameter('depth_yaml_path').value
        self._backup_dir = self.get_parameter('backup_dir').value
        self._min_solve = int(self.get_parameter('min_points_to_solve').value)

        self._marker_configs = self._load_marker_config(self._marker_cfg_path)
        self._calib_poses = self._load_calib_poses(self._poses_path)

        self._detector = MarkerDetector(self._marker_configs)
        self._data = DataCollector()
        self._still = StillnessDetector()

        # Depth estimator is imported lazily to keep import-time light and
        # to provide a clearer error if the module is missing.
        self._depth_estimator = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._K: Optional[np.ndarray] = None
        self._dist: Optional[np.ndarray] = None
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._latest_bgr: Optional[np.ndarray] = None
        self._latest_stamp = None
        self._frame_lock = threading.Lock()
        self._latest_he: Optional[HandEyeResult] = None
        self._latest_T_cam_base: Optional[np.ndarray] = None
        self._latest_depth = None
        self._sequence_lock = threading.Lock()

        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        info_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CompressedImage, self._image_topic,
            self._image_cb, img_qos, callback_group=self._cb_group)
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic,
            self._info_cb, info_qos, callback_group=self._cb_group)
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_cb, img_qos, callback_group=self._cb_group)

        self._pub_status = self.create_publisher(
            String, '/ba_calib/status', latched_qos)
        self._pub_datapoints = self.create_publisher(
            Int32, '/ba_calib/datapoints', 10)
        self._pub_reproj = self.create_publisher(
            Float32, '/ba_calib/reprojection_px', 10)
        self._pub_depth_rmse = self.create_publisher(
            Float32, '/ba_calib/depth_rmse_m', 10)
        self._pub_he = self.create_publisher(
            TransformStamped, '/ba_calib/hand_eye', latched_qos)
        self._pub_depth = self.create_publisher(
            Float32MultiArray, '/ba_calib/depth_params', latched_qos)
        self._pub_debug_image = self.create_publisher(
            Image, '/ba_calib/debug/image', 1)

        self.create_service(
            Trigger, '~/run_sequence',
            self._srv_run_sequence, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/recalibrate',
            self._srv_recalibrate, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/reset',
            self._srv_reset, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/save',
            self._srv_save, callback_group=self._cb_group)
        self.create_service(
            Trigger, '~/load',
            self._srv_load, callback_group=self._cb_group)

        self._runner = SequenceRunner(
            self,
            planning_group=self.get_parameter('planning_group').value,
            velocity_scaling=self.get_parameter('velocity_scaling').value,
            acceleration_scaling=self.get_parameter('acceleration_scaling').value,
        )

        self._publish_status('idle')
        self.get_logger().info(
            f'ba_autocalib ready. Markers: {list(self._detector._configs.keys())}. '
            f'Poses loaded: {len(self._calib_poses)}')

    # -----------------------------------------------------------------
    # Config loading
    # -----------------------------------------------------------------
    def _load_marker_config(self, path: str) -> List[MarkerConfig]:
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'marker_config_file not set or missing ({path!r}); '
                'using built-in defaults.')
            return _default_marker_configs()
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        configs = []
        for name, cfg in (data.get('markers') or {}).items():
            configs.append(MarkerConfig(
                name=name,
                tf_frame=cfg['tf_frame'],
                hsv_lower=tuple(cfg['hsv_lower']),
                hsv_upper=tuple(cfg['hsv_upper']),
                min_area_px=int(cfg.get('min_area_px', 30)),
                max_area_px=int(cfg.get('max_area_px', 5000)),
            ))
        if not configs:
            return _default_marker_configs()
        return configs

    def _load_calib_poses(self, path: str) -> List[CalibPose]:
        if not path or not os.path.isfile(path):
            self.get_logger().warn(
                f'calib_poses_file not set or missing ({path!r}); '
                'sequence service will be unavailable.')
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return [CalibPose(name=p['name'], joints=list(p['joints']))
                for p in (data.get('poses') or [])]

    # -----------------------------------------------------------------
    # ROS callbacks
    # -----------------------------------------------------------------
    def _info_cb(self, msg: CameraInfo) -> None:
        if self._K is not None:
            return
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        w, h = msg.width, msg.height
        self._K = K.copy()
        self._dist = np.zeros(5) if self._rectify else D.copy()
        if self._rectify:
            self._map_x, self._map_y = cv2.initUndistortRectifyMap(
                K, D, None, K, (w, h), cv2.CV_32FC1)
        self.get_logger().info(
            f'Got camera_info: {w}x{h}, fx={K[0,0]:.1f}, fy={K[1,1]:.1f}')
        if self._info_sub is not None:
            self.destroy_subscription(self._info_sub)
            self._info_sub = None

    def _image_cb(self, msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        if self._rectify and self._map_x is not None:
            bgr = cv2.remap(bgr, self._map_x, self._map_y,
                            cv2.INTER_LINEAR)
        with self._frame_lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp

    def _joint_cb(self, msg: JointState) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self._still.update(now, msg.velocity or [])

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------
    def _ensure_depth_loaded(self) -> None:
        if self._depth_estimator is not None:
            return
        from ba_depth_node.depth_estimator import DepthEstimator
        self.get_logger().info('Loading Depth-Anything-V2...')
        self._depth_estimator = DepthEstimator(
            model_id=self.get_parameter('depth_model_id').value,
            device=self.get_parameter('depth_device').value,
            log_fn=self.get_logger().info,
        )

    def _sample_once(self, pose_name: str) -> int:
        """Capture one image+depth, add a datapoint per detected marker."""
        if self._K is None:
            self.get_logger().warn('No camera_info yet; skipping sample')
            return 0
        with self._frame_lock:
            bgr = None if self._latest_bgr is None else self._latest_bgr.copy()
        if bgr is None:
            self.get_logger().warn('No image buffered yet; skipping sample')
            return 0

        self._ensure_depth_loaded()
        # DepthEstimator.estimate() returns float32 in [0, 1] directly —
        # we keep raw DA V2 values for consistency with the existing
        # depth_calibration.yaml schema (d_range in raw DA units).
        d_rel_map = self._depth_estimator.estimate(bgr)

        detections = self._detector.detect(bgr)
        added = 0
        now_s = self.get_clock().now().nanoseconds * 1e-9
        for name, det in detections.items():
            cfg = self._detector._configs[name]
            try:
                tf = self._tf_buffer.lookup_transform(
                    self._base_frame, cfg.tf_frame,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5))
            except LookupException as exc:
                self.get_logger().warn(
                    f'TF lookup base<-{cfg.tf_frame} failed: {exc}')
                continue
            P_base = np.array([
                tf.transform.translation.x,
                tf.transform.translation.y,
                tf.transform.translation.z,
            ])
            u, v = det.uv
            iu, iv = int(round(u)), int(round(v))
            h, w = d_rel_map.shape[:2]
            if not (0 <= iu < w and 0 <= iv < h):
                continue
            d = float(d_rel_map[iv, iu])
            dp = DataPoint(
                marker=name,
                P_base=P_base,
                uv=np.array([u, v]),
                d_rel=d,
                timestamp=now_s,
            )
            ok, reason = self._data.try_add(dp)
            if ok:
                added += 1
                self.get_logger().info(
                    f'  + {name} uv=({u:.1f},{v:.1f}) '
                    f'base=({P_base[0]:.3f},{P_base[1]:.3f},{P_base[2]:.3f}) '
                    f'd_rel={d:.3f}')
            else:
                self.get_logger().debug(f'  - {name} rejected: {reason}')

        self._pub_datapoints.publish(Int32(data=len(self._data)))
        self._publish_debug_image(bgr, detections)
        return added

    def _publish_debug_image(self, bgr: np.ndarray,
                              detections: Dict) -> None:
        reprojections = None
        if (self._latest_T_cam_base is not None and self._K is not None
                and detections):
            reprojections = {}
            for name, det in detections.items():
                cfg = self._detector._configs[name]
                try:
                    tf = self._tf_buffer.lookup_transform(
                        self._base_frame, cfg.tf_frame,
                        rclpy.time.Time())
                except LookupException:
                    continue
                P_base = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z, 1.0])
                P_cam = self._latest_T_cam_base @ P_base
                if P_cam[2] <= 0:
                    continue
                u = self._K[0, 0] * P_cam[0] / P_cam[2] + self._K[0, 2]
                v = self._K[1, 1] * P_cam[1] / P_cam[2] + self._K[1, 2]
                reprojections[name] = (u, v)

        overlay = self._detector.draw_overlay(bgr, detections, reprojections)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_rect'
        msg.height, msg.width = overlay.shape[:2]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = overlay.shape[1] * 3
        msg.data = overlay.tobytes()
        self._pub_debug_image.publish(msg)

    # -----------------------------------------------------------------
    # Solving
    # -----------------------------------------------------------------
    def _solve_and_publish(self) -> Optional[str]:
        if self._K is None:
            return 'no camera_info yet'
        points = self._data.snapshot()
        if len(points) < self._min_solve:
            return f'only {len(points)} points, need {self._min_solve}'
        try:
            he = solve_hand_eye(points, self._K, self._dist)
        except Exception as exc:
            return f'hand-eye solver failed: {exc}'
        self._latest_he = he
        self._latest_T_cam_base = he.T_cam_base
        self._pub_reproj.publish(Float32(data=he.median_px))
        self._publish_hand_eye_transform(he.T_cam_base)

        try:
            depth = solve_depth(points, he.T_cam_base)
        except Exception as exc:
            self.get_logger().warn(f'depth solver failed: {exc}')
            depth = None

        if depth is not None:
            self._latest_depth = depth
            self._pub_depth_rmse.publish(Float32(data=depth.rmse_m))
            self._pub_depth.publish(Float32MultiArray(
                data=[depth.a, depth.b, depth.rmse_m, float(depth.n_samples)]))

        status = 'good' if he.median_px < 3.0 and (
            depth is None or depth.rmse_m < 0.03) else 'degraded'
        self._publish_status(status)
        self.get_logger().info(
            f'Solved: reprojection_median={he.median_px:.2f}px '
            f'(n={he.n_points})'
            + ('' if depth is None else
               f', depth_rmse={depth.rmse_m*1000:.1f}mm '
               f'a={depth.a:.4f} b={depth.b:.4f}'))
        return None

    def _publish_hand_eye_transform(self, T_cam_base: np.ndarray) -> None:
        # /ba_calib/hand_eye publishes T_base_cam (camera origin in base frame)
        # to mirror the semantics of perception_pipeline.yaml's hand_eye_transform.
        T_base_cam = invert_transform(T_cam_base)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._base_frame
        t.child_frame_id = 'camera_rect'
        t.transform.translation.x = float(T_base_cam[0, 3])
        t.transform.translation.y = float(T_base_cam[1, 3])
        t.transform.translation.z = float(T_base_cam[2, 3])
        q = _mat_to_quat(T_base_cam[:3, :3])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self._pub_he.publish(t)

    def _publish_status(self, text: str) -> None:
        self._pub_status.publish(String(data=text))

    # -----------------------------------------------------------------
    # Services
    # -----------------------------------------------------------------
    def _srv_run_sequence(self, request, response):
        if not self._calib_poses:
            response.success = False
            response.message = 'No poses loaded (calib_poses_file missing)'
            return response
        if not self._sequence_lock.acquire(blocking=False):
            response.success = False
            response.message = 'A sequence is already running'
            return response
        # Fire and forget — worker releases the lock.
        thread = threading.Thread(
            target=self._run_sequence_worker, daemon=True)
        thread.start()
        response.success = True
        response.message = f'Sequence started with {len(self._calib_poses)} poses'
        return response

    def _run_sequence_worker(self) -> None:
        try:
            self._publish_status('collecting')
            settle = float(self.get_parameter('settle_s').value)
            visited = self._runner.run(
                self._calib_poses,
                self._per_pose_capture,
                settle_s=settle,
            )
            self.get_logger().info(
                f'Sequence done: visited {visited} poses, '
                f'{len(self._data)} datapoints buffered')
            self._publish_status('solving')
            err = self._solve_and_publish()
            if err:
                self.get_logger().error(f'Solve failed: {err}')
                self._publish_status('degraded')
        except Exception as exc:
            self.get_logger().error(
                f'Sequence worker crashed: {exc}\n{traceback.format_exc()}')
            self._publish_status('degraded')
        finally:
            self._sequence_lock.release()

    def _per_pose_capture(self, pose: CalibPose) -> None:
        # Wait briefly for stillness before sampling.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self._still.is_still(now_s=now):
                break
            time.sleep(0.05)
        added = self._sample_once(pose.name)
        self.get_logger().info(
            f'Pose {pose.name}: +{added} points '
            f'(total {len(self._data)})')

    def _srv_recalibrate(self, request, response):
        err = self._solve_and_publish()
        if err:
            response.success = False
            response.message = err
        else:
            response.success = True
            response.message = 'Solved from buffered data'
        return response

    def _srv_reset(self, request, response):
        n = len(self._data)
        self._data.clear()
        self._latest_he = None
        self._latest_T_cam_base = None
        self._pub_datapoints.publish(Int32(data=0))
        self._publish_status('idle')
        response.success = True
        response.message = f'Dropped {n} datapoints'
        return response

    def _srv_save(self, request, response):
        if self._latest_T_cam_base is None:
            response.success = False
            response.message = 'No hand-eye result yet — run /recalibrate first'
            return response
        if not self._perception_yaml or not os.path.isfile(
                self._perception_yaml):
            response.success = False
            response.message = (
                f'perception_yaml_path invalid: {self._perception_yaml!r}')
            return response
        try:
            he_backup, _ = write_hand_eye(
                self._perception_yaml, self._latest_T_cam_base,
                self._backup_dir)
            msg = [f'hand-eye written (backup: {he_backup})']
            if getattr(self, '_latest_depth', None) is not None and \
                    self._depth_yaml:
                d_backup = write_depth(
                    self._depth_yaml, self._latest_depth, self._backup_dir)
                if d_backup:
                    msg.append(f'depth written (backup: {d_backup})')
                else:
                    msg.append('depth written (new file)')
            response.success = True
            response.message = '; '.join(msg)
        except Exception as exc:
            response.success = False
            response.message = f'Save failed: {exc}'
        return response

    def _srv_load(self, request, response):
        # Reading back is easy; currently we just report where the files are.
        msg = []
        if self._perception_yaml and os.path.isfile(self._perception_yaml):
            msg.append(f'perception: {self._perception_yaml}')
        if self._depth_yaml and os.path.isfile(self._depth_yaml):
            msg.append(f'depth: {self._depth_yaml}')
        response.success = bool(msg)
        response.message = '; '.join(msg) or 'no YAMLs configured'
        return response


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _default_marker_configs() -> List[MarkerConfig]:
    return [
        MarkerConfig(
            name='grasp', tf_frame='marker_grasp',
            hsv_lower=(5, 150, 100), hsv_upper=(20, 255, 255)),
        MarkerConfig(
            name='forearm', tf_frame='marker_forearm',
            hsv_lower=(45, 100, 80), hsv_upper=(75, 255, 255)),
        MarkerConfig(
            name='base', tf_frame='marker_base',
            hsv_lower=(100, 150, 80), hsv_upper=(130, 255, 255)),
    ]


def _mat_to_quat(R: np.ndarray) -> tuple:
    """3x3 rotation matrix to quaternion (x, y, z, w)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return (float(qx), float(qy), float(qz), float(qw))


def main(args=None):
    rclpy.init(args=args)
    node = AutoCalibNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
