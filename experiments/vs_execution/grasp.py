import os
import time
import math

import bosdyn.client
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client import robot_command
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

from . import vs
from experiments.experiments_helpers.object_properties import OBJECTS
from bosdyn.api import geometry_pb2

try:
    from dotenv import load_dotenv
    load_dotenv()

except ImportError:
    pass


def power_on():
    print("Powering on robot...")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'
    print("Robot powered on!")


def setup_robot():
    power_on()
    print("Commanding robot to stand...")
    robot_command.blocking_stand(command_client, timeout_sec=10,
                                 params=RobotCommandBuilder.mobility_params(body_height=-1))
    print("Robot is standing!")
    print("Moving arm into position...")
    cmd = vs.vs_init_arm_gripper_pose_cmd(
        robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
    cmd_id = command_client.robot_command(cmd)
    robot_command.block_until_arm_arrives(command_client, cmd_id)
    print("Arm in position!")


def grasp(dist_to_floor, gripper_open_fraction, gripper_max_vel):
    # move the arm straight to the floor
    # Get robot state to compute frames
    robot_state = robot_state_client.get_robot_state()
    gpe_T_hand = frame_helpers.get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        frame_helpers.GROUND_PLANE_FRAME_NAME,
        frame_helpers.HAND_FRAME_NAME
    )
    odom_T_gpe = frame_helpers.get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        frame_helpers.ODOM_FRAME_NAME,
        frame_helpers.GROUND_PLANE_FRAME_NAME
    )
    # move the arm closer to the floor
    gpe_T_hand.z = dist_to_floor
    new_T = odom_T_gpe*gpe_T_hand
    cmd = RobotCommandBuilder.arm_pose_command(
        new_T.x, new_T.y, new_T.z, new_T.rot.w, new_T.rot.x, new_T.rot.y, new_T.rot.z, frame_helpers.ODOM_FRAME_NAME, 3)
    cmd_id = command_client.robot_command(cmd)
    robot_command.block_until_arm_arrives(command_client, cmd_id)

    # close the gripper
    cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
        gripper_open_fraction, max_vel=gripper_max_vel, disable_force_on_contact=True, max_torque=0.5)
    command_client.robot_command(cmd)
    time.sleep(2)


def carry():
    print("Moving arm into carrying position...")
    cmd = RobotCommandBuilder.arm_carry_command()
    cmd_id = command_client.robot_command(cmd)
    robot_command.block_until_arm_arrives(command_client, cmd_id)
    print("Arm in carrying position!")


def power_off():
    print("Powering robot off...")
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    print('Robot safely powered off!')


def grasp(duration_vs, x_off, y_off, z_off, yaw_off, ignore_obj_yaw, duration_ss, offset_ss, gripper_opening_fraction, gripper_max_vel, sleep_after_gripper, debug=False):
    if debug:
        input("Enter for grasping...")
    print('Grasping...')
    vs.vs(robot_state_client, command_client,
          duration_vs, x_off, y_off, z_off, yaw_off, ignore_obj_yaw)
    if debug:
        input('Enter to continue...')
    vs.vs_single_shot(robot_state_client, command_client,
                      duration_ss, offset_ss, x_off, y_off, z_off, yaw_off, ignore_obj_yaw)
    if debug:
        input('Enter to continue...')
    cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
        gripper_opening_fraction, max_vel=gripper_max_vel, disable_force_on_contact=True, max_torque=0.5)
    command_client.robot_command(cmd)
    time.sleep(sleep_after_gripper)
    print("Grasp done!")
    if debug:
        input('Enter to continue...')


if __name__ == '__main__':

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

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

        debug = False
        i = input('Grasp: p c cs cd t? ')

        setup_robot()

        if i == 'p':  # pen
            theta = math.radians(25) / 2
            offset_ss_Q = geometry_pb2.Quaternion(
                w=math.cos(theta), x=math.sin(theta), y=0, z=0)
            offset_ss = math_helpers.SE3Pose(x=0, y=0, z=0.075,
                                             rot=offset_ss_Q)
            grasp(2, 0, -0.01, 0.2, 0, False, 1.5,
                  offset_ss, 0.05, 3.0, 1, debug)

        if i == 'c':  # cup
            theta = math.radians(13) / 2
            offset_ss_Q = geometry_pb2.Quaternion(
                w=math.cos(theta), x=math.sin(theta), y=0, z=0)
            offset_ss = math_helpers.SE3Pose(x=-0.02, y=0, z=0.2,
                                             rot=offset_ss_Q)
            grasp(2, 0, 0.065, 0.4, 0, True, 3.5,
                  offset_ss, 0.05, 3.0, 1, debug)

        if i == 'cs':   # cup side
            theta = math.radians(-5) / 2
            offset_ss_Q = geometry_pb2.Quaternion(
                w=math.cos(theta), x=math.sin(theta), y=0, z=0)
            offset_ss = math_helpers.SE3Pose(x=0, y=0, z=0.275,
                                             rot=offset_ss_Q)
            grasp(2, 0, 0, 0.4, 0, False, 3.5, offset_ss, 0.55, 0.5, 2, debug)

        if i == 'cd':   # cup down
            theta = math.radians(-5) / 2
            offset_ss_Q = geometry_pb2.Quaternion(
                w=math.cos(theta), x=math.sin(theta), y=0, z=0)
            offset_ss = math_helpers.SE3Pose(x=0, y=0, z=0.223,
                                             rot=offset_ss_Q)
            grasp(1.5, 0, -0.01, 0.4, 0, True, 2,
                  offset_ss, 0.5, 0.5, 2, debug)

        if i == 't':  # transparent
            theta = math.radians(0) / 2
            offset_ss_Q = geometry_pb2.Quaternion(
                w=math.cos(theta), x=math.sin(theta), y=0, z=0)
            offset_ss = math_helpers.SE3Pose(x=0, y=0, z=0.275,
                                             rot=offset_ss_Q)
            grasp(2, 0.01, 0, 0.4, 0, True, 2,
                  offset_ss, 0.35, 0.5, 2.5, debug)

        carry()
        time.sleep(2)
        cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1, max_vel=1, disable_force_on_contact=True, max_torque=0.5)
        command_client.robot_command(cmd)
        power_off()
