"""Read / write calibration values to the canonical YAMLs.

- Hand-eye: config/perception_pipeline.yaml, key `hand_eye_transform`
- Depth:    config/depth_calibration.yaml (full rewrite, same schema as
  scripts/calibrate_depth.py)

Hand-eye uses regex-based in-place replacement to preserve the surrounding
comments and structure of the YAML. Depth uses a full PyYAML rewrite since
the file is purely calibration data.
"""

import datetime
import os
import re
import shutil
from typing import Tuple

import numpy as np
import yaml

from .solvers import DepthResult, invert_transform


_HAND_EYE_PATTERN = re.compile(
    r'(hand_eye_transform\s*:\s*)\[[^\]]*\]',
    re.DOTALL,
)


def backup(path: str, backup_dir: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base = os.path.basename(path)
    dst = os.path.join(backup_dir, f'{stamp}_{base}')
    shutil.copy2(path, dst)
    return dst


def format_hand_eye_block(T_base_cam: np.ndarray,
                          indent: str = '      ',
                          closing_indent: str = '    ') -> str:
    """Format a 4x4 matrix as the multi-line list used in the YAML.

    The default indentation matches the existing style in
    `config/perception_pipeline.yaml` (6 spaces for values, 4 for closer).
    """
    rows = []
    for i in range(4):
        vals = ', '.join(f'{T_base_cam[i, j]: .5f}' for j in range(4))
        rows.append(f'{indent}{vals},')
    body = '\n'.join(rows)
    return '[\n' + body + '\n' + closing_indent + ']'


def write_hand_eye(yaml_path: str, T_cam_base: np.ndarray,
                   backup_dir: str) -> Tuple[str, np.ndarray]:
    """Write T_base_cam (= inverse of T_cam_base) to `hand_eye_transform`.

    The key in perception_pipeline.yaml is defined as 'camera -> robot',
    i.e. P_robot = T @ P_cam. Our solver produces T_cam_base (robot -> cam),
    so we invert before writing.

    Returns (backup_path, T_base_cam).
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(yaml_path)
    T_base_cam = invert_transform(T_cam_base)
    block = format_hand_eye_block(T_base_cam)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not _HAND_EYE_PATTERN.search(content):
        raise RuntimeError(
            'hand_eye_transform key not found or not in bracketed-list form '
            f'in {yaml_path}')

    backup_path = backup(yaml_path, backup_dir)
    new_content = _HAND_EYE_PATTERN.sub(
        lambda m: m.group(1) + block, content, count=1)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return backup_path, T_base_cam


def write_depth(yaml_path: str, result: DepthResult,
                backup_dir: str) -> str:
    """Rewrite depth_calibration.yaml matching calibrate_depth.py schema."""
    if os.path.isfile(yaml_path):
        backup_path = backup(yaml_path, backup_dir)
    else:
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        backup_path = ''

    payload = {
        'model_type': 'inverse',
        'a': float(result.a),
        'b': float(result.b),
        'rmse_m': float(result.rmse_m),
        'n_samples': int(result.n_samples),
        'n_captures': int(result.n_samples),
        'created': datetime.datetime.now().isoformat(),
        'depth_range_m': [float(result.depth_range_m[0]),
                          float(result.depth_range_m[1])],
        'd_range': [float(result.d_range[0]), float(result.d_range[1])],
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return backup_path
