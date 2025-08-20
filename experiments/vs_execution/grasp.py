import os
import math
import time
import struct
from multiprocessing import shared_memory

from bosdyn.api import geometry_pb2
import bosdyn.client
from bosdyn.client import math_helpers
from bosdyn.client import frame_helpers
from bosdyn.client import robot_command
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient


SHM_NAME = 'shm_vs'


try:
    from dotenv import load_dotenv
    load_dotenv()

except ImportError:
    pass


def Q_down(yaw):
    half_y = math.radians(90) / 2.0
    qy = (math.cos(half_y), 0.0, math.sin(half_y), 0.0)

    half_z = math.radians(yaw) / 2.0
    qz = (math.cos(half_z), 0.0, 0.0, math.sin(half_z))

    qw = qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3]
    qx = qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2]
    qy_ = qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1]
    qz_ = qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0]
    return geometry_pb2.Quaternion(w=qw, x=qx, y=qy_, z=qz_)


def arm_pose_cmd_flat_body_T_yaw_hand(x, y, z, yaw, duration):
    flat_body_T_hand = geometry_pb2.Vec3(x=x, y=y, z=z)
    flat_body_Q_hand = Q_down(yaw)
    flat_body_tform_hand = geometry_pb2.SE3Pose(position=flat_body_T_hand,
                                                rotation=flat_body_Q_hand)

    transforms_snapshot = robot_state_client.get_robot_state(
    ).kinematic_state.transforms_snapshot
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


def init_arm_gripper_cmd():
    arm_cmd = arm_pose_cmd_flat_body_T_yaw_hand(0.75, 0, 0.25, 0, 2)
    gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
        1.0)
    cmd = RobotCommandBuilder.build_synchro_command(
        arm_cmd, gripper_cmd)
    return cmd


def power_on():
    print("Powering on robot...")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'
    print("Robot powered on!")


def power_off():
    print("Powering robot off...")
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    print('Robot safely powered off!')


def setup_robot():
    power_on()
    print("Commanding robot to stand...")
    robot_command.blocking_stand(command_client, timeout_sec=10,
                                 params=RobotCommandBuilder.mobility_params(body_height=-1))
    print("Robot is standing!")
    print("Moving arm into position...")
    cmd_id = command_client.robot_command(init_arm_gripper_cmd())
    robot_command.block_until_arm_arrives(command_client, cmd_id)
    print("Arm in position!")


def carry():
    print("Moving arm inty carrying position...")
    cmd = RobotCommandBuilder.arm_carry_command()
    cmd_id = command_client.robot_command(cmd)
    robot_command.block_until_arm_arrives(command_client, cmd_id)
    print("Arm in carrying position!")


def vs(z_offset, duration):
    print('VS...')

    t_start = time.time()
    t_now = time.time()

    while t_now - t_start < (duration - 0.2):

        x, y, z, yaw = struct.unpack('dddd', buffer[:32])
        z += z_offset

        cmd = arm_pose_cmd_flat_body_T_yaw_hand(
            x, y, z, yaw, duration - (t_now - t_start))
        cmd_id = command_client.robot_command(cmd)

        t_now = time.time()

    robot_command.block_until_arm_arrives(command_client, cmd_id)
    print('VS done!')


def grasp():
    cmd = RobotCommandBuilder.arm_wrench_command(
        0, 0, -2.5, 0, 0, 0, "flat_body", 3)
    cmd_id = command_client.robot_command(cmd)
    time.sleep(1)
    cmd = RobotCommandBuilder.claw_gripper_close_command(max_torque=1.75)
    command_client.robot_command(cmd)
    time.sleep(2)


ROBOT_IP = os.getenv("ROBOT_IP")

print("Establishing connection to spot...")
sdk = bosdyn.client.create_standard_sdk('vs_execution')
robot = sdk.create_robot(ROBOT_IP)
bosdyn.client.util.authenticate(robot)
robot.sync_with_directory()
robot.time_sync.wait_for_sync()
assert not robot.is_estopped(), 'Robot is estopped'
print("Connection to spot established!")

robot_state_client = robot.ensure_client(
    RobotStateClient.default_service_name)
command_client = robot.ensure_client(
    RobotCommandClient.default_service_name)
lease_client = robot.ensure_client(
    bosdyn.client.lease.LeaseClient.default_service_name)


shm = shared_memory.SharedMemory(name=SHM_NAME)
buffer = shm.buf


with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
    setup_robot()
    input('Enter to grasp...')
    print('Grasping...')
    vs(0.1, 10)
    grasp()
    print('Grasping done')
    carry()
    power_off()
