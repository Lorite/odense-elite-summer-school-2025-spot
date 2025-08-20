import argparse
import sys
import time
import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.robot_command import (
    RobotCommandClient, RobotCommandBuilder,
    blocking_stand, blocking_sit, block_until_arm_arrives
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (
    get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
)
from bosdyn.client.math_helpers import Quat, SE3Pose, SE3Velocity
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.util import seconds_to_timestamp

g_image_click = None
g_image_display = None
g_forward_distance = 0.5
g_last_distance_sent = None
cmd_response = None
g_height_offset = 0.0   # Adjustable Z offset in meters


def forward_trackbar(val):
    global g_forward_distance
    g_forward_distance = val / 100.0


def height_trackbar(val):
    global g_height_offset
    g_height_offset = (val - 50) / 100.0   # Centered at 0, range -0.5m .. +0.5m


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)


def detect_non_gray_objects(img, camera_source, z_offset):
    """Detect non-gray objects and back-project centroid to 3D using intrinsics."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 30, 200])
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    non_gray_mask = cv2.bitwise_not(gray_mask)

    contours, _ = cv2.findContours(non_gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largest_contour], -1, (0, 0, 255), 2)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(img, f"({cX}, {cY})", (cX + 10, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # --- Back-project to 3D using pinhole intrinsics ---
            if camera_source and camera_source.HasField("pinhole"):
                fx = camera_source.pinhole.intrinsics.focal_length.x
                fy = camera_source.pinhole.intrinsics.focal_length.y
                cx = camera_source.pinhole.intrinsics.principal_point.x
                cy = camera_source.pinhole.intrinsics.principal_point.y

                Z = 0.5 + z_offset   # base forward reach + slider offset
                X = (cX - cx) * Z / fx
                Y = (cY - cy) * Z / fy

                cv2.putText(img, f"3D: ({X:.2f}, {Y:.2f}, {Z:.2f}) m",
                            (cX + 10, cY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return img, X, Y


def extend_arm_sequence(robot, command_client, robot_state_client, image_client):
    robot.logger.info("Deploying arm...")
    robot_cmd = RobotCommandBuilder.arm_ready_command()
    cmd_id = command_client.robot_command(robot_cmd)
    block_until_arm_arrives(command_client, cmd_id)

    robot_state = robot_state_client.get_robot_state()
    odom_T_grav_body = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        GRAV_ALIGNED_BODY_FRAME_NAME,
    )

    grav_body_T_task = SE3Pose(x=0.5, y=0.0, z=0.1, rot=Quat())
    odom_T_task = odom_T_grav_body * grav_body_T_task

    wrist_tform_tool_neutral = SE3Pose(x=0.25, y=0.0, z=0.0, rot=Quat())
    wrist_tform_tool_down = SE3Pose(
        x=0.25, y=0.0, z=0.0, rot=Quat(w=0.7071, x=0.0, y=-0.7071, z=0.0)
    )

    times = [0.0, 3.0]
    neutral_pose = SE3Pose(x=0.0, y=0.0, z=0.0, rot=Quat())
    extended_pose = SE3Pose(x=0.5, y=0.0, z=0.1, rot=Quat())
    zero_vel = SE3Velocity(0, 0, 0, 0, 0, 0)
    max_acc = 0.5
    max_linear_vel = 0.3
    max_angular_vel = 0.5

    robot.logger.info("Moving arm to 50% forward position...")
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

    robot.logger.info("Rotating gripper downward...")
    se3_poses = [extended_pose.to_proto(), extended_pose.to_proto()]
    se3_velocities = [zero_vel.to_proto(), zero_vel.to_proto()]
    ref_time = seconds_to_timestamp(time.time() + 1.0)
    robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
        se3_poses=se3_poses,
        times=[0.0, 3.0],
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

    robot.logger.info("Opening gripper...")
    robot_cmd = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(robot_cmd)
    time.sleep(2)

    robot.logger.info("Switching to hand camera feed for 10 seconds...")
    window_name = "Hand Camera Feed"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Height (cm)", window_name, 50, 100, height_trackbar)

    start_time = time.time()
    while time.time() - start_time < 30:
        z_offset = g_height_offset
        adjusted_pose = SE3Pose(x=0.5, y=0.0, z=0.1 + z_offset, rot=Quat())
        se3_poses = [adjusted_pose.to_proto(), adjusted_pose.to_proto()]
        ref_time = seconds_to_timestamp(time.time() + 0.5)
        robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
            se3_poses=se3_poses,
            times=[0.0, 1.0],
            se3_velocities=[zero_vel.to_proto(), zero_vel.to_proto()],
            root_frame_name=ODOM_FRAME_NAME,
            root_tform_task=odom_T_task.to_proto(),
            wrist_tform_tool=wrist_tform_tool_down.to_proto(),
            max_acc=max_acc,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            ref_time=ref_time,
        )
        command_client.robot_command(robot_cmd)

        image_responses = image_client.get_image_from_sources(["hand_color_image"])
        if len(image_responses) == 1:
            image = image_responses[0]
            dtype = (
                np.uint16
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
                else np.uint8
            )
            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)

            # Non-gray object detection with intrinsics
            img, X, Y = detect_non_gray_objects(img, image.source, g_height_offset)
            
            # Display the image with detected object
            cv2.imshow(window_name, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # Move arm to detected object
            if X is not None and Y is not None:
                robot.logger.info(f"Detected object at pixel ({X:.2f}, {Y:.2f})")
                # the arm is already pointed down, so we can just move it forward in te hand frame. X is forward, Y is left/right
                adjusted_pose = SE3Pose(
                    x=0.4, y=(Y - img.shape[1] / 2) * 0.001, z=(X - img.shape[0] / 2) * 0.001 + z_offset,
                    rot=Quat()                    
                )
                se3_poses = [adjusted_pose.to_proto()]
                ref_time = seconds_to_timestamp(time.time() + 0.5)
                robot_cmd = RobotCommandBuilder.arm_cartesian_move_helper(
                    se3_poses=se3_poses,
                    times=[0.0],
                    se3_velocities=[zero_vel.to_proto()],
                    root_frame_name="hand", # "hand"
                )
                command_client.robot_command(robot_cmd)
                time.sleep(15)
                break
    cv2.destroyWindow(window_name)

    robot.logger.info("Closing gripper...")
    robot_cmd = RobotCommandBuilder.claw_gripper_close_command()
    command_client.robot_command(robot_cmd)
    time.sleep(2)

    robot.logger.info("Tucking arm...")
    robot_cmd = RobotCommandBuilder.arm_stow_command()
    cmd_id = command_client.robot_command(robot_cmd)
    block_until_arm_arrives(command_client, cmd_id)

    robot.logger.info("Commanding robot to sit...")
    blocking_sit(command_client, timeout_sec=10)


def walk_to_object_live(config):
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk("WalkToObjectClient")
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
    assert not robot.is_estopped(), "Robot is estopped. Please release E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."

        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)

        window_name = "Click to walk up to object"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, cv_mouse_callback)
        cv2.createTrackbar(
            "Forward (cm)", window_name, int(g_forward_distance * 100), 200, forward_trackbar
        )

        global g_image_display, cmd_response, g_last_distance_sent, g_image_click

        while True:
            image_responses = image_client.get_image_from_sources([config.image_source])
            if len(image_responses) != 1:
                continue
            image = image_responses[0]
            dtype = (
                np.uint16
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16
                else np.uint8
            )
            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
            g_image_display = img.copy()

            if g_image_click:
                cv2.line(
                    g_image_display,
                    (0, g_image_click[1]),
                    (img.shape[1], g_image_click[1]),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    g_image_display,
                    (g_image_click[0], 0),
                    (g_image_click[0], img.shape[0]),
                    (0, 255, 0),
                    2,
                )

                if cmd_response is None or g_last_distance_sent != g_forward_distance:
                    walk_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
                    offset_distance = wrappers_pb2.FloatValue(value=g_forward_distance)
                    walk_to = manipulation_api_pb2.WalkToObjectInImage(
                        pixel_xy=walk_vec,
                        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                        frame_name_image_sensor=image.shot.frame_name_image_sensor,
                        camera_model=image.source.pinhole,
                        offset_distance=offset_distance,
                    )
                    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
                        walk_to_object_in_image=walk_to
                    )
                    cmd_response = manipulation_api_client.manipulation_api_command(
                        manipulation_api_request=walk_to_request
                    )
                    g_last_distance_sent = g_forward_distance

                    time.sleep(2)
                    extend_arm_sequence(robot, command_client, robot_state_client, image_client)
                    g_image_click = None
                    break

            cv2.imshow(window_name, g_image_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        robot.logger.info("Powering off robot...")
        robot.power_off(cut_immediately=False, timeout_sec=20)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-i", "--image-source", help="Get image from source", default="frontleft_fisheye_image"
    )
    options = parser.parse_args()

    try:
        walk_to_object_live(options)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
