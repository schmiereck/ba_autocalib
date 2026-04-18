"""Drives the arm through a list of calibration poses via MoveIt.

Uses the same pattern as ba_perception_pipeline.goal_generator_node:
we send JointConstraint goals directly to /move_action. The 'arm' group
is 5-DOF (joint_0..joint_4); joint_5 / joint_5_mimic are ignored.

The runner is synchronous and thread-safe: call `run()` from a worker
thread while the node is spinning on a MultiThreadedExecutor.
"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from rclpy.node import Node


@dataclass
class CalibPose:
    name: str
    joints: List[float]   # joint_0..joint_4 (5 values)


class SequenceRunner:
    def __init__(self, node: Node,
                 planning_group: str = 'arm',
                 joint_tolerance: float = 0.01,
                 planning_time_s: float = 10.0,
                 velocity_scaling: float = 0.2,
                 acceleration_scaling: float = 0.2,
                 action_name: str = 'move_action'):
        self._node = node
        self._group = planning_group
        self._joint_tol = joint_tolerance
        self._plan_time = planning_time_s
        self._vel = velocity_scaling
        self._acc = acceleration_scaling
        self._client = ActionClient(node, MoveGroup, action_name)
        self._stop = threading.Event()
        self._active_gh = None
        self._active_lock = threading.Lock()

    def cancel(self) -> None:
        """Stop the sweep and cancel the in-flight MoveIt goal (if any).

        Safe to call from a signal handler / destroy_node: move_group runs
        in a different process, so the goal keeps executing unless we
        explicitly tell the action server to cancel it.
        """
        self._stop.set()
        with self._active_lock:
            gh = self._active_gh
        if gh is not None:
            try:
                gh.cancel_goal_async()
                self._node.get_logger().info(
                    'SequenceRunner: cancel sent to active MoveIt goal')
            except Exception as exc:
                self._node.get_logger().warn(
                    f'SequenceRunner: cancel failed: {exc}')

    def wait_for_server(self, timeout_s: float = 5.0) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def run(self, poses: List[CalibPose],
            per_pose_cb: Callable[[CalibPose, int, int], None],
            settle_s: float = 0.5,
            per_pose_timeout_s: float = 30.0,
            joint_names: Optional[List[str]] = None) -> int:
        if joint_names is None:
            joint_names = [f'joint_{i}' for i in range(5)]
        if not self.wait_for_server(timeout_s=5.0):
            self._node.get_logger().error('move_action server not available')
            return 0

        self._stop.clear()
        visited = 0
        total = len(poses)
        for idx, pose in enumerate(poses, start=1):
            if self._stop.is_set():
                self._node.get_logger().info(
                    f'SequenceRunner: stop requested, '
                    f'aborting after {visited}/{total} poses')
                break
            if len(pose.joints) != len(joint_names):
                self._node.get_logger().warn(
                    f'[{idx}/{total}] Pose {pose.name}: '
                    f'expected {len(joint_names)} joint values, '
                    f'got {len(pose.joints)}')
                continue
            joints_str = '[' + ', '.join(f'{v:+.3f}' for v in pose.joints) + ']'
            self._node.get_logger().info(
                f'[{idx}/{total}] Moving to pose {pose.name} '
                f'joints={joints_str}')
            ok = self._send_pose_sync(pose, joint_names, per_pose_timeout_s)
            if not ok:
                self._node.get_logger().warn(
                    f'Pose {pose.name}: MoveIt failure — skipping')
                continue
            time.sleep(settle_s)
            try:
                per_pose_cb(pose, idx, total)
            except Exception as exc:
                self._node.get_logger().error(
                    f'Pose {pose.name} callback raised: {exc}')
            visited += 1
        return visited

    def _send_pose_sync(self, pose: CalibPose, joint_names: List[str],
                        timeout_s: float) -> bool:
        constraints = Constraints()
        constraints.name = f'calib_{pose.name}'
        for name, value in zip(joint_names, pose.joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(value)
            jc.tolerance_above = self._joint_tol
            jc.tolerance_below = self._joint_tol
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self._group
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = self._plan_time
        goal_msg.request.max_velocity_scaling_factor = self._vel
        goal_msg.request.max_acceleration_scaling_factor = self._acc
        goal_msg.request.goal_constraints.append(constraints)
        goal_msg.planning_options.plan_only = False

        done = threading.Event()
        holder = {'ok': False}

        def on_result(fut):
            try:
                res = fut.result()
                holder['ok'] = bool(
                    res and res.result.error_code.val == 1)
            except Exception:
                holder['ok'] = False
            finally:
                done.set()

        def on_goal_response(fut):
            gh = fut.result()
            if gh is None or not gh.accepted:
                done.set()
                return
            with self._active_lock:
                self._active_gh = gh
            gh.get_result_async().add_done_callback(on_result)

        send_fut = self._client.send_goal_async(goal_msg)
        send_fut.add_done_callback(on_goal_response)

        try:
            if not done.wait(timeout_s):
                self._node.get_logger().warn(
                    f'Pose {pose.name}: timeout after {timeout_s}s')
                return False
            return holder['ok']
        finally:
            with self._active_lock:
                self._active_gh = None
