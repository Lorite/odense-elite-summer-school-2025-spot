import os
import math
import time
import ast
import struct

from multiprocessing import shared_memory

import bosdyn.client
from bosdyn.client import math_helpers
from bosdyn.api import geometry_pb2
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.robot_command import RobotCommandClient, block_until_arm_arrives, RobotCommandBuilder, blocking_stand, spot_command_pb2
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, ODOM_FRAME_NAME


def q_straight_down(yaw_rad=0):
    half_y = math.radians(90) / 2.0
    qy = (math.cos(half_y), 0.0, math.sin(half_y), 0.0)

    half_z = yaw_rad / 2.0
    qz = (math.cos(half_z), 0.0, 0.0, math.sin(half_z))

    qw = qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3]
    qx = qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2]
    qy_ = qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1]
    qz_ = qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0]

    return geometry_pb2.Quaternion(w=qw, x=qx, y=qy_, z=qz_)


def cmd_flat_body(x, y, z, angle, t):
    # -8

    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    flat_body_Q_hand = q_straight_down(math.radians(angle))
    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                            rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * \
        math_helpers.SE3Pose.from_proto(flat_body_T_hand)

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
        odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, t)

    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
        1.0)

    command = RobotCommandBuilder.build_synchro_command(
        gripper_command, arm_command)

    return command


try:
    from dotenv import load_dotenv
    load_dotenv()

except ImportError:
    pass

ROBOT_IP = os.getenv("ROBOT_IP")


def main():
    global robot_state_client, command_client

    shm = shared_memory.SharedMemory(name='shm_goal')
    buffer = shm.buf

    try:
        print("Establishing connection to spot...")
        sdk = bosdyn.client.create_standard_sdk('vs_execution')
        robot = sdk.create_robot(ROBOT_IP)
        bosdyn.client.util.authenticate(robot)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()
        print("Connection to spot established!")

        assert not robot.is_estopped(), 'Robot is estopped'

        robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name)
        command_client = robot.ensure_client(
            RobotCommandClient.default_service_name)

        lease_client = robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

            print("Powering on robot...")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), 'Robot power on failed.'
            print("Robot powered on!")

            print("Commanding robot to stand...")
            blocking_stand(command_client, timeout_sec=10,
                           params=RobotCommandBuilder.mobility_params(body_height=-1))
            print("Robot is standing!")

            print("Moving arm into position...")
            cmd_id = command_client.robot_command(
                cmd_flat_body(0.75, 0, 0.25, 0, 2))
            block_until_arm_arrives(command_client, cmd_id)
            print("Arm in position!")

            input("Enter to grasp")
            print("Grasping...")

            t_start = time.time()
            t_now = time.time()

            duration = 3

            while t_now - t_start < duration:

                x, y, z, angle = struct.unpack('dddd', buffer[:32])
                z += 0.10
                cmd_id = command_client.robot_command(cmd_flat_body(
                    x, y, z, angle, duration - (t_now - t_start)+0.5))

                t_now = time.time()

            block_until_arm_arrives(command_client, cmd_id)

            print("Grasp Done")
            # input("Enter...")

            cmd = RobotCommandBuilder.arm_wrench_command(
                0, 0, -2.5, 0, 0, 0, "flat_body", 3)
            cmd_id = command_client.robot_command(cmd)

            time.sleep(1)

            cmd = RobotCommandBuilder.claw_gripper_close_command(max_torque=1.75)
            command_client.robot_command(cmd)

            time.sleep(2)

            # input("Enter...")

            cmd = RobotCommandBuilder.arm_carry_command()
            command_client.robot_command(cmd)

            input("Enter...")

            print("Powering robot off...")
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), 'Robot power off failed.'
            robot.logger.info('Robot safely powered off!')

    finally:
        shm._flags = 0
        shm.close()


if __name__ == "__main__":
    main()
