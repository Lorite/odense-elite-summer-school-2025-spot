# Copyright (c) 2023 Boston Dynamics, Inc.
# Integrated body-camera click → hand-camera centering → grasp script.

import argparse
import sys
import time
import cv2
import numpy as np
import os

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client import image as bd_image
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
# already imported above, also add this constant for readability:
GAB_FRAME_NAME = frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME


g_image_click = None
g_image_display = None


def verify_estop(robot):
    """Verify the robot is not estopped."""
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception("Robot is estopped. Release E-Stop before running.")


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    if g_image_display is None:
        return
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # draw crosshair
        color = (30, 30, 30)
        thickness = 2
        height, width = clone.shape[:2]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(param, clone)


def _infer_depth_source(visual_source: str) -> str:
    """### NEW: Map visual source to the aligned depth source."""
    if visual_source.endswith('_fisheye_image'):
        return visual_source.replace('_fisheye_image', '_depth_in_visual_frame')
    return visual_source + '_depth_in_visual_frame'


def _decode_visual_to_numpy(image_proto):
    """### NEW: Decode a visual frame for display."""
    dtype = (np.uint16 if image_proto.shot.image.pixel_format ==
             image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8)
    buf = np.frombuffer(image_proto.shot.image.data, dtype=dtype)
    if image_proto.shot.image.format == image_pb2.Image.FORMAT_RAW:
        return buf.reshape(image_proto.shot.image.rows, image_proto.shot.image.cols)
    return cv2.imdecode(buf, -1)


def _decode_depth_to_numpy(depth_proto):
    """### NEW: Decode a depth frame to uint16 array (rows x cols)."""
    return np.frombuffer(depth_proto.shot.image.data, dtype=np.uint16).reshape(
        depth_proto.shot.image.rows, depth_proto.shot.image.cols
    )


def _unproject_with_pinhole(depth_resp, u, v, depth_m):
    """### NEW: Unproject (u,v,depth_m) to camera XYZ using pinhole intrinsics."""
    intr = depth_resp.source.pinhole.intrinsics
    fx, fy = intr.focal_length.x, intr.focal_length.y
    cx, cy = intr.principal_point.x, intr.principal_point.y
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return float(x), float(y), float(z)


def pixel_to_xyz_from_depth_value(depth_resp, u, v, depth_m, out_frame=VISION_FRAME_NAME):
    """### NEW: Use a known depth value (in meters) at (u,v) to compute XYZ in out_frame."""
    # Unproject to camera frame
    x_cam, y_cam, z_cam = _unproject_with_pinhole(depth_resp, u, v, depth_m)

    # camera -> out_frame (VISION or BODY)
    T_out_cam = frame_helpers.get_a_tform_b(
        depth_resp.shot.transforms_snapshot, out_frame, depth_resp.shot.frame_name_image_sensor
    )
    x, y, z = T_out_cam.transform_point(x_cam, y_cam, z_cam)
    return np.asarray([x, y, z]), out_frame


def pixel_to_xyz_in_frame(image_client, visual_source, click_uv, out_frame=VISION_FRAME_NAME):
    """
    (Kept for completeness) Convert a pixel (u,v) from a visual stream to a 3D point in out_frame,
    by fetching a depth frame once. Not used in the synced path.
    """
    u, v = int(click_uv[0]), int(click_uv[1])
    depth_source = _infer_depth_source(visual_source)
    depth_resp = image_client.get_image_from_sources([depth_source])[0]

    depth_u16 = _decode_depth_to_numpy(depth_resp)[v, u]
    if depth_u16 == 0 or depth_u16 == 65535:
        raise ValueError(f"No valid depth at pixel {u,v}.")

    depth_scale = getattr(depth_resp.source, 'depth_scale', 1000.0)
    depth_m = float(depth_u16) / float(depth_scale)

    x_cam, y_cam, z_cam = bd_image.pixel_to_camera_space(depth_resp.source, u, v, depth_m)

    T_out_cam = frame_helpers.get_a_tform_b(
        depth_resp.shot.transforms_snapshot, out_frame, depth_resp.shot.frame_name_image_sensor
    )
    xyz_out = T_out_cam.transform_point(float(x_cam), float(y_cam), float(z_cam))
    return np.asarray(xyz_out), out_frame, depth_resp.shot.transforms_snapshot

def _quat_gripper_down_in(root_frame_name=GAB_FRAME_NAME):
    """
    Quaternion that orients the hand so its +X axis points 'down' in the root frame
    (i.e., along -Z of that frame). ey sets wrist roll (here aligned with +X_root).
    """
    ex = np.array([0.0, 0.0, -1.0])  # +X_hand -> -Z_root (down)
    ey = np.array([1.0, 0.0,  0.0])  # choose roll
    ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    R = np.column_stack([ex, ey, ez])  # maps hand->root
    return math_helpers.Quat.from_matrix(R)



def move_hand_to_point_with_gravity_offset(command_client, state_client, xyz_vision,
                                           down_offset_m=0.30, seconds=5.0):
    """
    Fetch full frame tree, transform VISION -> GRAV_ALIGNED_BODY (or BODY fallback),
    apply +h on Z (Z up), and move gripper-down. 'seconds' controls how slow the move is.
    """
    # Full snapshot (has GRAV_ALIGNED_BODY)
    snapshot_full = state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Prefer GRAV_ALIGNED_BODY; fallback to BODY
    T_root_vision = frame_helpers.get_a_tform_b(snapshot_full,
                                                frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME,
                                                frame_helpers.VISION_FRAME_NAME)
    frame_name = frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
    if T_root_vision is None:
        T_root_vision = frame_helpers.get_a_tform_b(snapshot_full,
                                                    frame_helpers.BODY_FRAME_NAME,
                                                    frame_helpers.VISION_FRAME_NAME)
        frame_name = frame_helpers.BODY_FRAME_NAME

    # Transform point into chosen root frame
    x_v, y_v, z_v = map(float, xyz_vision)
    x_r, y_r, z_r = T_root_vision.transform_point(x_v, y_v, z_v)

    # Target above the point by h (Z up)
    z_target = z_r + float(down_offset_m)

    # Gripper-down orientation in the same root frame
    ex = np.array([0.0, 0.0, -1.0])  # +X_hand -> -Z_root
    ey = np.array([1.0, 0.0,  0.0])  # choose wrist roll
    ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)
    q_down = math_helpers.Quat.from_matrix(np.column_stack([ex, ey, ez]))

    final = math_helpers.SE3Pose(x_r, y_r, z_target, q_down).to_proto()
    final_cmd = RobotCommandBuilder.arm_pose_command_from_pose(final, frame_name, seconds=seconds)

    return command_client.robot_command(final_cmd)


def get_click_with_synced_depth(image_client, visual_source, window_name):
    """
    ### NEW: Display a *synced* visual stream while fetching the aligned depth *for the same frame*.
    Returns (click_uv, visual_resp_cache, depth_resp_cache, depth_img_cache) for the exact shown frame.
    """
    global g_image_click, g_image_display
    g_image_click = None

    depth_source = _infer_depth_source(visual_source)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, cv_mouse_callback, param=window_name)

    visual_resp_cache = None
    depth_resp_cache = None
    depth_img_cache = None

    while g_image_click is None:
        # Fetch both sources together every cycle so they are time-aligned.
        resps = image_client.get_image_from_sources([visual_source, depth_source])
        if len(resps) != 2:
            continue

        # Identify which is which
        if resps[0].source.name == visual_source:
            v_resp, d_resp = resps[0], resps[1]
        else:
            v_resp, d_resp = resps[1], resps[0]

        # Decode for display/cache
        vis_img = _decode_visual_to_numpy(v_resp)
        depth_img = _decode_depth_to_numpy(d_resp)

        # Update caches tied to this displayed frame
        visual_resp_cache = v_resp
        depth_resp_cache = d_resp
        depth_img_cache = depth_img
        g_image_display = vis_img

        # Show the visual frame
        cv2.imshow(window_name, g_image_display)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # q or ESC
            print('Quit.')
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return g_image_click, visual_resp_cache, depth_resp_cache, depth_img_cache


def median_depth_at_uv(image_client, depth_source, u, v, first_depth_resp, first_depth_img, samples=1, delay_s=0.0):
    """
    ### NEW: Return a *temporal median* depth (in meters) at (u,v) over `samples` frames.
    - Uses the cached (synced) depth for the first sample (no extra wait).
    - If samples == 1, returns immediately.
    - Ignores invalid raw values (0 or 65535).
    """
    # Prepare vectors to collect raw values
    raw_vals = []

    # First sample: the cached one (no wait)
    try:
        raw0 = int(first_depth_img[v, u])
        if raw0 not in (0, 65535):
            raw_vals.append(raw0)
    except Exception:
        pass

    # Additional samples (if requested)
    for _ in range(max(0, int(samples) - 1)):
        d_resp = image_client.get_image_from_sources([depth_source])[0]
        d_img = _decode_depth_to_numpy(d_resp)
        raw = int(d_img[v, u])
        if raw not in (0, 65535):
            raw_vals.append(raw)
        if delay_s > 0.0:
            time.sleep(delay_s)

    if len(raw_vals) == 0:
        raise ValueError(f"No valid depth at ({u},{v}) across {samples} sample(s).")

    # Median of raw values → meters using the first response's scale (all should match)
    raw_med = float(np.median(np.asarray(raw_vals, dtype=np.float32)))
    depth_scale = getattr(first_depth_resp.source, 'depth_scale', 1000.0)
    depth_m = raw_med / float(depth_scale)
    return depth_m


def add_grasp_constraint(config, grasp):
    use_vector_constraint = config.force_top_down_grasp or config.force_horizontal_grasp
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
    if use_vector_constraint:
        if config.force_top_down_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        elif config.force_horizontal_grasp:
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    elif config.force_squeeze_grasp:
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def setup_robot_and_clients(config):
    """Create SDK, authenticate, time sync, verify estop, and get required clients."""
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspCenteringClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    return robot, lease_client, image_client, manipulation_api_client, command_client, state_client


def power_on_and_stand(robot, command_client):
    """Power on and make the robot stand."""
    print("Powering on robot...")
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), "Power on failed."

    print("Standing...")
    blocking_stand(command_client, timeout_sec=10)


def click_and_compute_xyz(image_client, visual_source, depth_samples):
    """Get a user click on a synced visual frame and compute the 3D point in VISION using cached depth."""
    click_xy, vis_resp, depth_resp, depth_img = get_click_with_synced_depth(
        image_client, visual_source, "Body Camera Click"
    )
    u, v = int(click_xy[0]), int(click_xy[1])

    depth_source = _infer_depth_source(visual_source)
    depth_m = median_depth_at_uv(
        image_client=image_client,
        depth_source=depth_source,
        u=u,
        v=v,
        first_depth_resp=depth_resp,
        first_depth_img=depth_img,
        samples=max(1, int(depth_samples)),
        delay_s=0.0,
    )

    # Compute XYZ in VISION frame using the cached depth response
    xyz_vision, frame = pixel_to_xyz_from_depth_value(depth_resp, u, v, depth_m, out_frame=VISION_FRAME_NAME)
    return click_xy, xyz_vision, frame


def execute_pick_sequence(command_client, state_client, xyz_vision, down_offset):
    """Move arm near the target with gravity offset and preshape gripper as in the original sequence."""
    cmd_id = move_hand_to_point_with_gravity_offset(
        command_client,
        state_client=state_client,
        xyz_vision=xyz_vision,
        down_offset_m=down_offset,
        seconds=2.0,
    )
    block_until_arm_arrives(command_client, cmd_id)
    print(f"Moved hand to gravity-offset target (h={down_offset:.3f} m).")

    command_client.robot_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(
            open_fraction=0.9, max_vel=0.5, max_torque=5.0
        )
    )
    time.sleep(3)
    print("Opened gripper to 90%.")

    cmd_id = move_hand_to_point_with_gravity_offset(
        command_client,
        state_client=state_client,
        xyz_vision=xyz_vision,
        down_offset_m=0.02,
        seconds=2.0,
    )
    block_until_arm_arrives(command_client, cmd_id)
    print("Moved hand to gravity-offset target (h=0.02 m).")

    command_client.robot_command(
        RobotCommandBuilder.claw_gripper_open_fraction_command(
            open_fraction=0.5, max_vel=0.4, max_torque=0.01
        )
    )
    time.sleep(3)
    print("Opened gripper to 50%.")

    cmd_id = move_hand_to_point_with_gravity_offset(
        command_client,
        state_client=state_client,
        xyz_vision=xyz_vision,
        down_offset_m=0.3,
        seconds=2.0,
    )
    block_until_arm_arrives(command_client, cmd_id)
    print("Moved hand to target with down offset of 0.3 m.")

    # carry the object
    carry(command_client)


def arm_object_grasp_with_centering(config):
    robot, lease_client, image_client, manipulation_api_client, command_client, state_client = setup_robot_and_clients(config)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        power_on_and_stand(robot, command_client)

        print("Click on the object in the camera window. Press 'q' in the window or Ctrl+C here to quit.")
        try:
            while True:
                try:
                    click_xy, xyz_vision, frame = click_and_compute_xyz(
                        image_client, config.image_source, config.depth_samples
                    )
                    print(
                        f"Click {click_xy} → 3D ({frame}): [{xyz_vision[0]:.3f}, {xyz_vision[1]:.3f}, {xyz_vision[2]:.3f}] m"
                    )

                    try:
                        execute_pick_sequence(
                            command_client, state_client, xyz_vision, down_offset=config.down_offset
                        )
                    except Exception as e:
                        print(f"[Warn] Arm move failed: {e}")
                except Exception as e:
                    print(f"[Warn] Could not compute 3D for click: {e}")
        except KeyboardInterrupt:
            print("\nStopping click-to-3D test (keyboard interrupt).")


def carry(command_client):
    print("Moving arm into carrying position...")
    cmd = RobotCommandBuilder.arm_carry_command()
    cmd_id = command_client.robot_command(cmd)
    block_until_arm_arrives(command_client, cmd_id)
    print("Arm in carrying position!")


def power_off(robot):
    print("Powering robot off...")
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    print('Robot safely powered off!')


def get_hostname_from_env_or_args(options):
    """Resolve hostname: prefer CLI option, else .env (ROBOT_IP or HOSTNAME). Exit if missing."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    hostname = getattr(options, 'hostname', None)
    if hostname:
        return hostname

    env_host = os.getenv('ROBOT_IP') or os.getenv('HOSTNAME')
    if env_host:
        return env_host

    print('Error: hostname not provided and not found in .env (ROBOT_IP or HOSTNAME).')
    sys.exit(1)


def ensure_hostname_cli_arg():
    """Ensure a hostname positional arg is present by reading .env (ROBOT_IP or HOSTNAME)."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    # If user already provided a positional hostname (non-flag), do nothing
    if any(not arg.startswith('-') for arg in sys.argv[1:]):
        return
    env_host = os.getenv('ROBOT_IP') or os.getenv('HOSTNAME')
    if env_host:
        # Append as positional argument to satisfy bosdyn base arguments
        sys.argv.append(env_host)


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-i', '--image-source', help='Body camera source', default='frontleft_fisheye_image')
    parser.add_argument('-hi', '--hand-image-source', help='Hand camera source', default='hand_color_image')
    parser.add_argument('-t', '--force-top-down-grasp', action='store_true')
    parser.add_argument('-f', '--force-horizontal-grasp', action='store_true')
    parser.add_argument('-s', '--force-squeeze-grasp', action='store_true')
    parser.add_argument('--depth-samples', type=int, default=3,
                        help='### NEW: Number of depth frames to temporally sample per click (median). 1 = no extra wait.')
    parser.add_argument('--down-offset', type=float, default=0.30,
                        help='Vertical (gravity) offset h in meters. Target Z becomes z + h in GRAV_ALIGNED_BODY.')

    # Inject hostname from .env if not provided on CLI to satisfy required arg
    ensure_hostname_cli_arg()

    options = parser.parse_args()

    # Resolve and normalize hostname via helper (keeps existing CLI value)
    options.hostname = get_hostname_from_env_or_args(options)

    if sum([options.force_top_down_grasp, options.force_horizontal_grasp, options.force_squeeze_grasp]) > 1:
        print('Error: Cannot force more than one type of grasp.')
        sys.exit(1)

    try:
        arm_object_grasp_with_centering(options)
        return True
    except Exception as exc:
        print(f"Exception: {exc}")
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
