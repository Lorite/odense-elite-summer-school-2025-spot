import argparse
import sys
import time
import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import Quat, SE3Pose, SE3Velocity
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import build_image_request

from bosdyn.util import seconds_to_timestamp


def stand_extend_arm_then_sit(config):
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk("StandExtendArmSit")
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    assert robot.has_arm(), "Robot requires an arm to run this example."
    assert not robot.is_estopped(), "Robot is estopped. Please clear estop before running."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        image_client = robot.ensure_client('image')

        # Stand up
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Ready the arm (deploy it)
        robot_cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)

        # Get robot state to compute frames
        robot_state = robot_state_client.get_robot_state()
        odom_T_grav_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME,
        )

        # Define task frame 0.75m forward of robot body, slight z offset (0.1m)
        grav_body_T_task = SE3Pose(x=0.75, y=0.0, z=0.1, rot=Quat())

        odom_T_task = odom_T_grav_body * grav_body_T_task

        # Wrist to tool transform - neutral (facing forward)
        wrist_tform_tool_neutral = SE3Pose(x=0.25, y=0.0, z=0.0, rot=Quat())

        # Wrist to tool transform - facing downward (-90 deg pitch)
        wrist_tform_tool_down = SE3Pose(
            x=0.25,
            y=0.0,
            z=0.0,
            rot=Quat(w=0.7071, x=0.0, y=-0.7071, z=0.0)  # approx -90 deg pitch
        )

        times = [0.0, 3.0]  # 3 seconds to move

        neutral_pose = SE3Pose(x=0.0, y=0.0, z=0.0, rot=Quat())
        extended_pose = SE3Pose(x=0.75, y=0.0, z=0.1, rot=Quat())

        zero_vel = SE3Velocity(0, 0, 0, 0, 0, 0)

        max_acc = 0.5
        max_linear_vel = 0.3
        max_angular_vel = 0.5

        # Move arm from neutral to extended position with gripper facing forward
        robot.logger.info("Moving arm to extended forward position...")
        se3_poses = [neutral_pose.to_proto(), extended_pose.to_proto()]
        se3_velocities = [zero_vel.to_proto(), zero_vel.to_proto()]
        ref_time = seconds_to_timestamp(time.time() + 1.0)

        robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
            se3_poses=se3_poses,
            times=times,
            se3_velocities=se3_velocities,
            root_frame_name=ODOM_FRAME_NAME,
            root_tform_task=odom_T_task.to_proto(),
            wrist_tform_tool=wrist_tform_tool_neutral.to_proto(),
            max_acc=max_acc,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            ref_time=ref_time,
        )
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)
        robot.logger.info("Arm extended forward.")

        # Rotate gripper downward (pitch -90 deg) while holding extension
        robot.logger.info("Rotating gripper downward...")
        se3_poses = [extended_pose.to_proto(), extended_pose.to_proto()]  # same position, change orientation
        se3_velocities = [zero_vel.to_proto(), zero_vel.to_proto()]
        times = [0.0, 3.0]
        ref_time = seconds_to_timestamp(time.time() + 1.0)

        robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
            se3_poses=se3_poses,
            times=times,
            se3_velocities=se3_velocities,
            root_frame_name=ODOM_FRAME_NAME,
            root_tform_task=odom_T_task.to_proto(),
            wrist_tform_tool=wrist_tform_tool_down.to_proto(),
            max_acc=max_acc,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            ref_time=ref_time,
        )
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)
        robot.logger.info("Gripper rotated downward.")

        # Open gripper
        robot.logger.info("Opening gripper...")
        cmd_id = command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(3)  # wait for gripper to open

        # **Stand longer with arm extended and gripper open**
        stand_time = 1000  # seconds to stay in this position
        robot.logger.info(f"Standing with arm extended and gripper open for {stand_time} seconds...")
        time.sleep(stand_time)

        # Return arm to neutral position and wrist facing forward
        robot.logger.info("Returning arm and wrist to neutral position...")
        times = [0.0, 3.0]
        se3_poses = [extended_pose.to_proto(), neutral_pose.to_proto()]
        se3_velocities = [zero_vel.to_proto(), zero_vel.to_proto()]
        ref_time = seconds_to_timestamp(time.time() + 1.0)

        robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
            se3_poses=se3_poses,
            times=times,
            se3_velocities=se3_velocities,
            root_frame_name=ODOM_FRAME_NAME,
            root_tform_task=odom_T_task.to_proto(),
            wrist_tform_tool=wrist_tform_tool_neutral.to_proto(),
            max_acc=max_acc,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            ref_time=ref_time,
        )
        cmd_id = command_client.robot_command(robot_cmd)
        block_until_arm_arrives(command_client, cmd_id)
        robot.logger.info("Arm and wrist returned to neutral.")

        # Sit down safely
        robot.logger.info("Powering off (sit down)...")
        robot.power_off(cut_immediately=False, timeout_sec=20)
        robot.logger.info("Robot safely powered off.")


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()

    try:
        stand_extend_arm_then_sit(options)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
