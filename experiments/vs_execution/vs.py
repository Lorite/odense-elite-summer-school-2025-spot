import os
import tempfile
import math
import time
import struct
import mmap

from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client import frame_helpers
from bosdyn.client import robot_command
from bosdyn.client.robot_command import RobotCommandBuilder

MMAP_FILENAME = os.path.join(tempfile.gettempdir(), "mmap_vs.bin")
MMAP_SIZE = 32


CAMERA_Q_HAND = geometry_pb2.Quaternion(
    w=0.5377, x=0.4592, y=-0.4592, z=0.5377)
CAMERA_TFORM_HAND = math_helpers.SE3Pose(x=0.020, y=0.033, z=0.053,
                                         rot=CAMERA_Q_HAND)

DURATION_LAST_TRAJECTORY = 0.2


def Q_down(yaw):
    w1, x1, y1, z1 = 0, -math.sqrt(2)/2, math.sqrt(2)/2, 0
    theta = math.radians(-yaw)
    w2 = math.cos(theta/2)
    x2 = 0
    y2 = 0
    z2 = math.sin(theta/2)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return geometry_pb2.Quaternion(w=w, x=x, y=y, z=z)


def T_yaw(x, y, z, yaw):
    Q = geometry_pb2.Quaternion(w=math.cos(math.radians(
        yaw/2)), x=0, y=0, z=math.sin(math.radians(yaw/2)))
    tform = math_helpers.SE3Pose(x=x, y=y, z=z, rot=Q)
    return tform


def arm_pose_cmd_flat_body_T_yaw_camera(transforms_snapshot, x, y, z, yaw, duration, offset=None):
    flat_body_Q_camera = Q_down(yaw)
    flat_body_tform_camera = math_helpers.SE3Pose(x=x, y=y, z=z,
                                                  rot=flat_body_Q_camera)

    if offset is not None:
        flat_body_tform_camera = flat_body_tform_camera * offset

    flat_body_tform_hand = flat_body_tform_camera * CAMERA_TFORM_HAND
    flat_body_tform_hand = flat_body_tform_hand.to_proto()

    odom_tform_flat_body = frame_helpers.get_a_tform_b(transforms_snapshot,
                                                       frame_helpers.ODOM_FRAME_NAME,
                                                       frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_tform_hand = odom_tform_flat_body * \
        math_helpers.SE3Pose.from_proto(flat_body_tform_hand)

    arm_pose_command = RobotCommandBuilder \
        .arm_pose_command(odom_tform_hand.x,
                          odom_tform_hand.y,
                          odom_tform_hand.z,
                          odom_tform_hand.rot.w,
                          odom_tform_hand.rot.x,
                          odom_tform_hand.rot.y,
                          odom_tform_hand.rot.z,
                          frame_helpers.ODOM_FRAME_NAME,
                          duration)
    return arm_pose_command


def vs_init_arm_gripper_pose_cmd(robot_transforms_snapshot, duration=1, x=0.75, y=0, z=0.25, yaw=0):
    """
    Robot command to bring arm/gripper in valid initial position for visual servoing 
    (camera looking straight down, gripper open).

    Args:
        robot_transforms_snapshot: Transforms snapshot from robot_state.kinematic_state
        duration: Duration of the trajectory
        x, y, z: Translation w.r.t. flat_body frame
        yaw: yaw w.r.t flat_body frame

    Returns:
        The command
    """
    arm_cmd = arm_pose_cmd_flat_body_T_yaw_camera(
        robot_transforms_snapshot, x, y, z, 0, duration)
    gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
        1.0)
    cmd = RobotCommandBuilder.build_synchro_command(
        arm_cmd, gripper_cmd)
    return cmd


def vs(state_client, command_client, duration, x_off=0, y_off=0, z_off=0.2, yaw_off=0, ignore_obj_yaw=False):
    """
    Visual Servoing. Uses perception output to move endeffector to predefined pose to the object (with 
    camera looking straight down). Assumes that camera is initally looking straigt down with object in
    frame and detected by running perception.

    Args:
        state_client: Robot state client.
        command_client: Robot command client.
        duration: Duration of the movement.
        x_off, y_off, z_off: Target translation camera pose <-> object pose.
        yaw_off: Target yaw camera pose <-> object pose (camera pose <-> flat_body if ignore_obj_yaw)
        ignore_obj_yaw: Ignore object yaw, just use object translation; see yaw_off
    """
    with open(MMAP_FILENAME, 'r+b') as f:
        shm = mmap.mmap(f.fileno(), MMAP_SIZE, access=mmap.ACCESS_READ)

    offset = T_yaw(x_off, y_off, z_off, -yaw_off)
    offset = offset.inverse()

    t_start = time.time()
    t_now = time.time()

    while t_now - t_start < (duration - DURATION_LAST_TRAJECTORY):

        x, y, z, yaw = struct.unpack('dddd', shm[:32])
        if ignore_obj_yaw:
            yaw = 0

        transforms_snapshot = state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot
        cmd = arm_pose_cmd_flat_body_T_yaw_camera(
            transforms_snapshot, x, y, z, yaw, duration - (t_now - t_start), offset)
        cmd_id = command_client.robot_command(cmd)

        t_now = time.time()

    robot_command.block_until_arm_arrives(command_client, cmd_id)
