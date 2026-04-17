"""Quality-filtered ring buffer for calibration data points.

Feeds solvers with DataPoint tuples. Rejects points that would dilute the
solve (arm moving, low-quality marker, redundant pose, bad depth).
"""

from collections import deque
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .solvers import DataPoint


class DataCollector:
    def __init__(self, max_points: int = 300,
                 pose_diversity_m: float = 0.03,
                 d_rel_min: float = 0.2,
                 d_rel_max: float = 0.95):
        self._points: deque = deque(maxlen=max_points)
        self._pose_diversity_m = pose_diversity_m
        self._d_rel_min = d_rel_min
        self._d_rel_max = d_rel_max

    def __len__(self) -> int:
        return len(self._points)

    def clear(self) -> None:
        self._points.clear()

    def snapshot(self) -> List[DataPoint]:
        return list(self._points)

    def snapshot_for_marker(self, name: str) -> List[DataPoint]:
        return [p for p in self._points if p.marker == name]

    def try_add(self, candidate: DataPoint) -> Tuple[bool, str]:
        """Returns (accepted, reason)."""
        if not (self._d_rel_min <= candidate.d_rel <= self._d_rel_max):
            return False, f'd_rel={candidate.d_rel:.3f} out of range'
        if not self._is_diverse(candidate):
            return False, 'pose not diverse enough'
        self._points.append(candidate)
        return True, 'accepted'

    def _is_diverse(self, candidate: DataPoint) -> bool:
        for p in self._points:
            if p.marker != candidate.marker:
                continue
            if np.linalg.norm(p.P_base - candidate.P_base) < \
                    self._pose_diversity_m:
                return False
        return True


class StillnessDetector:
    """Detects arm stillness from JointState velocity stream."""

    def __init__(self, velocity_threshold: float = 0.01,
                 window_s: float = 0.2):
        self._v_thresh = velocity_threshold
        self._window_s = window_s
        self._buf: deque = deque()

    def update(self, stamp_s: float, velocities: Iterable[float]) -> None:
        vmax = max((abs(v) for v in velocities), default=0.0)
        self._buf.append((stamp_s, vmax))
        # Drop entries older than window.
        while self._buf and stamp_s - self._buf[0][0] > self._window_s:
            self._buf.popleft()

    def is_still(self, now_s: Optional[float] = None) -> bool:
        if not self._buf:
            return False
        if now_s is not None:
            if now_s - self._buf[-1][0] > 0.5:
                return False
            if now_s - self._buf[0][0] < self._window_s * 0.9:
                return False
        return all(v < self._v_thresh for _, v in self._buf)
