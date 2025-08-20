# Copyright (c) 2023 Boston Dynamics, Inc.
# Dual front-camera click → compute 3D (from cached aligned depth) → arm sequence
# with an interactive object selector (Cup/Box/Pen) that adjusts grasp parameters.

import argparse
import sys
import time
import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client import image as bd_image
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers

# Gravity-aligned frame (Z up)
GAB_FRAME_NAME = frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME

# -----------------------
# Dual-window click state
# -----------------------
g_dual_click = {"which": None, "uv": None}
g_display_left = None
g_display_right = None


def verify_estop(robot):
    """Verify the robot is not estopped."""
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        raise Exception("Robot is estopped. Release E-Stop before running.")


def cv_mouse_callback_dual(event, x, y, flags, param):
    """param is the window tag: 'left' or 'right'"""
    global g_dual_click, g_display_left, g_display_right

    disp = g_display_left if param == 'left' else g_display_right
    if disp is None:
        return

    clone = disp.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_dual_click["which"] = param
        g_dual_click["uv"] = (x, y)
    else:
        # draw crosshair
        color = (30, 30, 30)
        thickness = 2
        h, w = clone.shape[:2]
        cv2.line(clone, (0, y), (w, y), color, thickness)
        cv2.line(clone, (x, 0), (x, h), color, thickness)
        win_name = "Front Left" if param == 'left' else "Front Right"
        cv2.imshow(win_name, clone)


def _infer_depth_source(visual_source: str) -> str:
    """Map visual source to the aligned depth source."""
    if visual_source.endswith('_fisheye_image'):
        return visual_source.replace('_fisheye_image', '_depth_in_visual_frame')
    return visual_source + '_depth_in_visual_frame'


def _decode_visual_to_numpy(image_proto):
    """Decode a visual frame for display."""
    dtype = (np.uint16 if image_proto.shot.image.pixel_format ==
             image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8)
    buf = np.frombuffer(image_proto.shot.image.data, dtype=dtype)
    if image_proto.shot.image.format == image_pb2.Image.FORMAT_RAW:
        return buf.reshape(image_proto.shot.image.rows, image_proto.shot.image.cols)
    return cv2.imdecode(buf, -1)


def _decode_depth_to_numpy(depth_proto):
    """Decode a depth frame to uint16 array (rows x cols)."""
    return np.frombuffer(depth_proto.shot.image.data, dtype=np.uint16).reshape(
        depth_proto.shot.image.rows, depth_proto.shot.image.cols
    )


def _unproject_with_pinhole(depth_resp, u, v, depth_m):
    """Unproject (u,v,depth_m) to camera XYZ using pinhole intrinsics."""
    intr = depth_resp.source.pinhole.intrinsics
    fx, fy = intr.focal_length.x, intr.focal_length.y
    cx, cy = intr.principal_point.x, intr.principal_point.y
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m
    return float(x), float(y), float(z)


def pixel_to_xyz_from_depth_value(depth_resp, u, v, depth_m, out_frame=VISION_FRAME_NAME):
    """Use a known depth value (in meters) at (u,v) to compute XYZ in out_frame."""
    x_cam, y_cam, z_cam = _unproject_with_pinhole(depth_resp, u, v, depth_m)
    T_out_cam = frame_helpers.get_a_tform_b(
        depth_resp.shot.transforms_snapshot, out_frame, depth_resp.shot.frame_name_image_sensor
    )
    x, y, z = T_out_cam.transform_point(x_cam, y_cam, z_cam)
    return np.asarray([x, y, z]), out_frame


def _quat_gripper_down_in(root_frame_name=GAB_FRAME_NAME, roll_rad=0.0, base_axis='x'):
    """
    Hand +X points 'down' (along -Z_root). Roll the wrist around that down axis by roll_rad.
    base_axis chooses the 0° reference for the jaw direction: 'x' or 'y' (root-frame axis).
    """
    ex = np.array([0.0, 0.0, -1.0], dtype=float)  # +X_hand -> -Z_root

    # Choose zero-roll reference axis in root frame (horizontal)
    if base_axis.lower() == 'y':
        ey0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        ey0 = np.array([1.0, 0.0, 0.0], dtype=float)

    # Project to plane orthogonal to ex
    ey0 = ey0 - np.dot(ey0, ex) * ex
    ey0 /= np.linalg.norm(ey0)

    # Rodrigues rotation: rotate ey0 around ex by roll_rad
    k = ex
    c = np.cos(roll_rad); s = np.sin(roll_rad)
    ey = ey0 * c + np.cross(k, ey0) * s + k * (np.dot(k, ey0)) * (1 - c)

    # Complete right-handed basis
    ez = np.cross(ex, ey); ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex); ey /= np.linalg.norm(ey)

    R = np.column_stack([ex, ey, ez])
    return math_helpers.Quat.from_matrix(R)


def move_hand_to_point_with_gravity_offset(command_client, state_client, xyz_vision,
                                           down_offset_m=0.30, seconds=5.0,
                                           roll_deg=0.0, base_axis='x'):
    """
    Use full robot frame tree to convert VISION → GAB (or BODY fallback),
    then move to (x,y,z + h) with gripper facing down and a given wrist roll.
    """
    snapshot_full = state_client.get_robot_state().kinematic_state.transforms_snapshot

    # Prefer GAB; fallback to BODY if needed
    T_root_vision = frame_helpers.get_a_tform_b(
        snapshot_full, GAB_FRAME_NAME, VISION_FRAME_NAME
    )
    frame_name = GAB_FRAME_NAME
    if T_root_vision is None:
        T_root_vision = frame_helpers.get_a_tform_b(
            snapshot_full, frame_helpers.BODY_FRAME_NAME, VISION_FRAME_NAME
        )
        frame_name = frame_helpers.BODY_FRAME_NAME
        print("[Warn] Falling back to BODY (Z may not be gravity-aligned).")

    x_v, y_v, z_v = map(float, xyz_vision)
    x_r, y_r, z_r = T_root_vision.transform_point(x_v, y_v, z_v)
    z_target = z_r + float(down_offset_m)  # “above” = +Z in GAB

    q_down = _quat_gripper_down_in(frame_name, np.deg2rad(roll_deg), base_axis=base_axis)

    # Pre-approach and final, using 'seconds' to control speed
    pre = math_helpers.SE3Pose(x_r, y_r, z_target + 0.10, q_down).to_proto()
    pre_cmd = RobotCommandBuilder.arm_pose_command_from_pose(pre, frame_name, seconds=seconds)

    final = math_helpers.SE3Pose(x_r, y_r, z_target, q_down).to_proto()
    final_cmd = RobotCommandBuilder.arm_pose_command_from_pose(final, frame_name, seconds=seconds)

    command_client.robot_command(pre_cmd)
    command_client.robot_command(final_cmd)


def get_click_with_synced_depth_dual(image_client, left_visual_source, right_visual_source):
    """
    Show two windows (front-left & front-right). Each refresh, fetch visual+aligned depth for both.
    Wait until user clicks in either window, then return the chosen side's cached frames.
    Returns:
        which: 'left' or 'right'
        click_uv: (u, v)
        visual_resp, depth_resp, depth_img: caches for the chosen camera
        visual_source: the chosen visual image source name
    """
    global g_dual_click, g_display_left, g_display_right
    g_dual_click = {"which": None, "uv": None}

    left_depth_source = _infer_depth_source(left_visual_source)
    right_depth_source = _infer_depth_source(right_visual_source)

    cv2.namedWindow("Front Left")
    cv2.namedWindow("Front Right")
    cv2.setMouseCallback("Front Left", cv_mouse_callback_dual, param='left')
    cv2.setMouseCallback("Front Right", cv_mouse_callback_dual, param='right')

    cache = {
        "left": {"v": None, "d": None, "img_depth": None},
        "right": {"v": None, "d": None, "img_depth": None},
    }

    while g_dual_click["which"] is None:
        # Left: fetch visual+depth
        res_l = image_client.get_image_from_sources([left_visual_source, left_depth_source])
        if len(res_l) == 2:
            v_l = res_l[0] if res_l[0].source.name == left_visual_source else res_l[1]
            d_l = res_l[1] if res_l[0].source.name == left_visual_source else res_l[0]
            img_l = _decode_visual_to_numpy(v_l)
            dep_l = _decode_depth_to_numpy(d_l)
            cache["left"].update({"v": v_l, "d": d_l, "img_depth": dep_l})
            g_display_left = img_l

        # Right: fetch visual+depth
        res_r = image_client.get_image_from_sources([right_visual_source, right_depth_source])
        if len(res_r) == 2:
            v_r = res_r[0] if res_r[0].source.name == right_visual_source else res_r[1]
            d_r = res_r[1] if res_r[0].source.name == right_visual_source else res_r[0]
            img_r = _decode_visual_to_numpy(v_r)
            dep_r = _decode_depth_to_numpy(d_r)
            cache["right"].update({"v": v_r, "d": d_r, "img_depth": dep_r})
            g_display_right = img_r

        # Show both (if available)
        if g_display_left is not None:
            cv2.imshow("Front Left", g_display_left)
        if g_display_right is not None:
            cv2.imshow("Front Right", g_display_right)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:  # ESC or q
            print("Quit.")
            sys.exit(0)

    cv2.destroyWindow("Front Left")
    cv2.destroyWindow("Front Right")

    which = g_dual_click["which"]
    click_uv = g_dual_click["uv"]
    chosen_cache = cache[which]
    if chosen_cache["v"] is None or chosen_cache["d"] is None:
        raise RuntimeError("No cached frames for the selected camera.")

    visual_source = left_visual_source if which == 'left' else right_visual_source
    return which, click_uv, chosen_cache["v"], chosen_cache["d"], chosen_cache["img_depth"], visual_source


def median_depth_at_uv(image_client, depth_source, u, v, first_depth_resp, first_depth_img, samples=1, delay_s=0.0):
    """
    Return a temporal median depth (in meters) at (u,v) over `samples` frames.
    Uses the cached (synced) depth as the first sample (no extra wait).
    Ignores invalid raw values (0 or 65535).
    """
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

    raw_med = float(np.median(np.asarray(raw_vals, dtype=np.float32)))
    depth_scale = getattr(first_depth_resp.source, 'depth_scale', 1000.0)
    depth_m = raw_med / float(depth_scale)
    return depth_m


# --------------------------
# Grasp profiles (C/B/P)
# --------------------------
GRASP_PROFILES = {
    # Hover safely higher, close gently (leave some opening)
    "C": {"name": "cup", "hover_offset": 0.30, "contact_offset": 0.03, "close_fraction": 0.5, "close_torque": 0.01},
    # Standard box: moderate hover, standard contact, full close
    "B": {"name": "box", "hover_offset": 0.25, "contact_offset": 0.02, "close_fraction": 0.4, "close_torque": 0.01},
    # Slim pen: hover modestly, contact very close, full close, but you may need lower torque
    "P": {"name": "pen", "hover_offset": 0.20, "contact_offset": 0.008, "close_fraction": 0.0, "close_torque": 5.0},
}


def ask_object_profile():
    """Prompt the user to choose C/B/P; keep asking until valid."""
    prompt = "Object to grasp? [C=cup, B=box, P=pen]: "
    while True:
        choice = input(prompt).strip().upper()
        if choice in GRASP_PROFILES:
            prof = GRASP_PROFILES[choice].copy()
            prof["key"] = choice
            print(f"Selected: {prof['name']}  "
                  f"(hover={prof['hover_offset']} m, contact={prof['contact_offset']} m, "
                  f"close_fraction={prof['close_fraction']}, torque={prof['close_torque']})")
            return prof
        print("Please type C, B, or P.")


def arm_object_grasp_with_centering(config):
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('ArmObjectGraspCenteringClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
    verify_estop(robot)

    # Pick grasp profile interactively
    grasp_profile = ask_object_profile()

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        print("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Power on failed."

        print("Standing...")
        blocking_stand(command_client, timeout_sec=10)

        print("Click in either window (Front Left / Front Right). Press 'q' or 'Esc' to quit.")
        try:
            while True:
                which, click_xy, vis_resp, depth_resp, depth_img, visual_source = \
                    get_click_with_synced_depth_dual(
                        image_client, config.left_image_source, config.right_image_source
                    )
                u, v = int(click_xy[0]), int(click_xy[1])

                # Bounds check
                h, w = depth_img.shape[:2]
                if not (0 <= u < w and 0 <= v < h):
                    print(f"[Warn] Click {click_xy} out of bounds {w}x{h} on {which} camera.")
                    continue

                try:
                    depth_source = _infer_depth_source(visual_source)
                    depth_m = median_depth_at_uv(
                        image_client=image_client,
                        depth_source=depth_source,
                        u=u, v=v,
                        first_depth_resp=depth_resp,
                        first_depth_img=depth_img,
                        samples=max(1, int(config.depth_samples)),
                        delay_s=0.0
                    )

                    # Compute XYZ from the cached depth (no refetch)
                    xyz_vision, frame = pixel_to_xyz_from_depth_value(
                        depth_resp, u, v, depth_m, out_frame=VISION_FRAME_NAME
                    )
                    print(f"[{which.upper()}] Click {click_xy} → 3D ({frame}): "
                          f"[{xyz_vision[0]:.3f}, {xyz_vision[1]:.3f}, {xyz_vision[2]:.3f}] m")

                    # ---- Sequence using grasp profile ----
                    move_seconds = 3.0   # increase to slow down further
                    settle = 0.3

                    # 1) Hover above target by profile hover_offset
                    move_hand_to_point_with_gravity_offset(
                        command_client, state_client, xyz_vision,
                        down_offset_m=grasp_profile["hover_offset"], seconds=move_seconds,
                        roll_deg=config.roll_deg, base_axis=config.roll_base_axis
                    )
                    time.sleep(move_seconds + settle)

                    # 2) Open gripper fully
                    command_client.robot_command(
                        RobotCommandBuilder.claw_gripper_open_fraction_command(
                            open_fraction=1.0, max_vel=0.5, max_torque=5.0
                        )
                    )
                    time.sleep(0.5)

                    # 3) Descend near contact by profile contact_offset
                    move_hand_to_point_with_gravity_offset(
                        command_client, state_client, xyz_vision,
                        down_offset_m=grasp_profile["contact_offset"], seconds=move_seconds,
                        roll_deg=config.roll_deg, base_axis=config.roll_base_axis
                    )
                    time.sleep(move_seconds + settle)

                    # 4) Close according to profile (close_fraction & torque)
                    command_client.robot_command(
                        RobotCommandBuilder.claw_gripper_open_fraction_command(
                            open_fraction=float(grasp_profile["close_fraction"]),
                            max_vel=0.6,
                            max_torque=float(grasp_profile["close_torque"])
                        )
                    )
                    time.sleep(1.2)

                    # 5) Lift back up to at least the hover height (use max of hover/contact/lift)
                    lift_offset = max(grasp_profile["hover_offset"], 0.3)
                    move_hand_to_point_with_gravity_offset(
                        command_client, state_client, xyz_vision,
                        down_offset_m=lift_offset, seconds=move_seconds,
                        roll_deg=config.roll_deg, base_axis=config.roll_base_axis
                    )
                    time.sleep(move_seconds + settle)

                except Exception as e:
                    print(f"[Warn] Could not compute/move for click {click_xy} on {which} camera: {e}")

        except KeyboardInterrupt:
            print("\nStopping click-to-3D test (keyboard interrupt).")


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)

    # Dual camera sources
    parser.add_argument('--left-image-source',  default='frontleft_fisheye_image',
                        help='Left front visual stream.')
    parser.add_argument('--right-image-source', default='frontright_fisheye_image',
                        help='Right front visual stream.')
    # (Optional) deprecated single camera flag; if provided, use as left
    parser.add_argument('-i', '--image-source', help='[Deprecated] Single camera source; used as left if provided.')

    # Depth sampling + default hover (may be overridden by profile)
    parser.add_argument('--depth-samples', type=int, default=3,
                        help='Number of depth frames to temporally median at the clicked pixel. 1 = no extra wait.')

    # Wrist roll
    parser.add_argument('--roll-deg', type=float, default=-90.0,
                        help='Wrist roll about gravity-down (degrees). Try 0, 90, 180, 270.')
    parser.add_argument('--roll-base-axis', choices=['x', 'y'], default='x',
                        help='Zero-roll reference axis in the root frame (affects jaw direction).')

    options = parser.parse_args()

    # Backward-compat for -i/--image-source
    if options.image_source:
        options.left_image_source = options.image_source

    try:
        arm_object_grasp_with_centering(options)
        return True
    except Exception as exc:
        print(f"Exception: {exc}")
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
