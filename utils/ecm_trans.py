"""
Utilities to convert an orientation (rotation matrix or quaternion)
into camera-style look_at and up vectors.

Assumptions:
- The camera forward axis is local -Z.
- The camera up axis is local +Y.
"""

from typing import Dict, Union
import numpy as np
from scipy.spatial.transform import Rotation


def _format_vec(vec: np.ndarray) -> Dict[str, float]:
    """Format a 3D vector as dict with keys x, y, z."""
    return {"x": float(vec[0]), "y": float(vec[1]), "z": float(vec[2])}


def rotation_to_look_up(rot_mat: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Convert a 3x3 rotation matrix to look_at and up vectors.

    Args:
        rot_mat: (3, 3) rotation matrix.

    Returns:
        dict with keys 'look_at' and 'up', each a dict {x, y, z}.
    """
    if rot_mat.shape != (3, 3):
        raise ValueError("rot_mat must be shape (3,3)")

    forward_local = np.array([-1, 0.0, 0])  # camera forward in local frame
    up_local = np.array([0.0, 0.0, 1.0])        # camera up in local frame

    look_at_world = rot_mat @ forward_local
    up_world = rot_mat @ up_local

    #round to 3 decimal places
    look_at_world = np.round(look_at_world, 1)
    up_world = np.round(up_world, 1)
    print(np.dot(look_at_world, up_world))

    return {"look_at": _format_vec(look_at_world),
            "up": _format_vec(up_world)}


def quaternion_to_look_up(quat: Union[np.ndarray, list, tuple]) -> Dict[str, Dict[str, float]]:
    """
    Convert a quaternion to look_at and up vectors.

    Args:
        quat: Quaternion, length 4. Accepts xyzw or wxyz ordering.

    Returns:
        dict with keys 'look_at' and 'up', each a dict {x, y, z}.
    """
    if len(quat) != 4:
        raise ValueError("quat must have 4 elements")

    quat = np.asarray(quat, dtype=float)

    # Detect ordering: if first element likely w, swap to xyzw
    # if abs(quat[0]) > 0.5:  # treat as wxyz
    #     quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
    # else:  # assume xyzw
    #     quat_xyzw = quat

    rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    #r = Rotation.from_euler("xyz", [0.7778962255, 0, -3.1415848732], degrees=False).as_matrix()
    # F = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    # T_CN_CV2 = F.dot(rot_mat)
    return rotation_to_look_up(rot_mat)


if __name__ == "__main__":
    # Example: identity orientation -> look_at (0,0,-1), up (0,1,0)
    # identity_rot = np.eye(3)
    # print(rotation_to_look_up(identity_rot))

    # # Example: rotate 30 deg around Y axis
    # r = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    # print(rotation_to_look_up(r))

    quat_xyzw = [-0.6437051794527742,
             -0.6424324828237596,
             -0.2936977949411906,
              0.29439047573801475]
    print(quaternion_to_look_up(quat_xyzw))

    # quat_xyzw = [0.033833643560676994,
    #          -0.0005958685063654413,
    #          -0.3586525580031255,
    #          0.9328575840619762]
    # quat = Rotation.from_quat(quat_xyzw).as_quat()
    # print(quat)
    # rpy = Rotation.from_quat(quat).as_euler("xyz", degrees=False)
    # print(rpy)


