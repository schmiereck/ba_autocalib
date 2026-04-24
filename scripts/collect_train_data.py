#!/usr/bin/env python3
"""Collect training data for marker detection using Pairwise Interpolation.

Drives the arm through randomized poses by interpolating between known-good
visible calibration poses. This ensures the gripper stays within the FOV.
"""

import os
import time
import random
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import CompressedImage
from rclpy.time import Time
import yaml

from ba_autocalib.sequence_runner import SequenceRunner, CalibPose

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('train_data_collector')
        self._cb_group = ReentrantCallbackGroup()
        
        self.declare_parameter('output_dir', '../../ba_calib_model_train/dataset/raw')
        self.declare_parameter('num_poses', 200)
        self.declare_parameter('poses_file', '../config/calib_poses.yaml')
        
        self._output_dir = self.get_parameter('output_dir').value
        os.makedirs(self._output_dir, exist_ok=True)
        
        # Load seed poses from calib_poses.yaml
        poses_path = self.get_parameter('poses_file').value
        with open(poses_path, 'r') as f:
            data = yaml.safe_load(f)
            self._seeds = [np.array(p['joints']) for p in data.get('poses', [])]
        
        if len(self._seeds) < 2:
            raise RuntimeError("Need at least 2 poses in calib_poses.yaml for interpolation!")

        self.get_logger().info(f'Collector initialized with {len(self._seeds)} anchor poses.')
            
        self._runner = SequenceRunner(self, velocity_scaling=0.4, acceleration_scaling=0.4)
        self._latest_bgr = None
        self._latest_stamp = None
        self._frame_lock = threading.Lock()
        
        self.create_subscription(
            CompressedImage, '/ba_overview_camera/image_raw/compressed',
            self._image_cb, 10, callback_group=self._cb_group)

    def _image_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None: return
        with self._frame_lock:
            self._latest_bgr = bgr
            self._latest_stamp = msg.header.stamp

    def run(self):
        if not self._runner.wait_for_server():
            self.get_logger().error('MoveIt not available!')
            return

        num_to_collect = self.get_parameter('num_poses').value
        self.get_logger().info(f'Starting collection of {num_to_collect} poses...')
        
        collected = 0
        while collected < num_to_collect and rclpy.ok():
            # 1. Pick TWO random seed poses
            p1, p2 = random.sample(self._seeds, 2)
            
            # 2. Linear Interpolation between them
            alpha = random.random()
            mixed_joints = p1 * alpha + p2 * (1.0 - alpha)
            
            # 3. Add a small local jitter (+/- 0.05 rad approx 3 deg)
            jitter = np.random.uniform(-0.05, 0.05, size=5)
            target_joints = (mixed_joints + jitter).tolist()
            
            pose = CalibPose(name=f'train_{collected:04d}', joints=target_joints)
            
            self.get_logger().info(f'[{collected+1}/{num_to_collect}] Moving to interpolated pose...')
            ok = self._runner._send_pose_sync(pose, [f'joint_{i}' for i in range(5)], timeout_s=15.0)
            
            if ok:
                time.sleep(1.0) # Settle
                
                # 4. Capture fresh image
                last_stamp_ns = 0
                with self._frame_lock:
                    if self._latest_stamp is not None:
                        last_stamp_ns = Time.from_msg(self._latest_stamp).nanoseconds
                
                img_ok = False
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    with self._frame_lock:
                        if self._latest_stamp is not None:
                            curr_ns = Time.from_msg(self._latest_stamp).nanoseconds
                            if curr_ns > last_stamp_ns:
                                bgr = self._latest_bgr.copy()
                                img_ok = True
                                break
                    time.sleep(0.02)
                
                if img_ok:
                    fname = os.path.join(self._output_dir, f'img_{collected:04d}.jpg')
                    cv2.imwrite(fname, bgr)
                    collected += 1
                    self.get_logger().info(f'Saved {fname}')
            else:
                self.get_logger().warn('Move failed, trying another combination...')

def main():
    rclpy.init()
    node = DataCollectionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
