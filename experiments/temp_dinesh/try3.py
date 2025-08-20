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
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, blocking_sit
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2

g_image_click = None
g_cursor_pos = None
g_cursor_cam = None
g_forward_distance = 0.5
g_last_distance_sent = None
cmd_response = None
clicked_camera_index = None

def forward_trackbar(val):
    global g_forward_distance
    g_forward_distance = val / 100.0

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)

def dual_mouse_callback(event, x, y, flags, param):
    global g_image_click, clicked_camera_index, g_cursor_pos, g_cursor_cam
    imgs = param
    if event == cv2.EVENT_MOUSEMOVE:
        if x < imgs[0].shape[1]:
            g_cursor_cam = 0
            g_cursor_pos = (x, y)
        else:
            g_cursor_cam = 1
            g_cursor_pos = (x - imgs[0].shape[1], y)

    if event == cv2.EVENT_LBUTTONUP:
        if x < imgs[0].shape[1]:
            clicked_camera_index = 0
            g_image_click = (x, y)
        else:
            clicked_camera_index = 1
            g_image_click = (x - imgs[0].shape[1], y)

def add_base_arguments(parser):
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--verbose", action="store_true")

def walk_to_object_live(config):
    global g_forward_distance, g_last_distance_sent, cmd_response, g_image_click, clicked_camera_index

    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk("WalkToObjectClient")
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please release E-Stop."

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."

        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)

        # --- Single camera walk to object ---
        window_name = "Click on object to walk"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, cv_mouse_callback)
        cv2.createTrackbar("Forward (cm)", window_name, int(g_forward_distance * 100), 200, forward_trackbar)

        while True:
            image_responses = image_client.get_image_from_sources([config.image_source])
            if len(image_responses) != 1:
                continue
            image = image_responses[0]
            dtype = np.uint16 if image.shot.image.pixel_format == 2 else np.uint8
            img = np.frombuffer(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == 0:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
            g_image_display = img.copy()

            if g_image_click:
                cv2.line(g_image_display, (0, g_image_click[1]), (img.shape[1], g_image_click[1]), (0, 255, 0), 2)
                cv2.line(g_image_display, (g_image_click[0], 0), (g_image_click[0], img.shape[0]), (0, 255, 0), 2)

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
                    walk_to_request = manipulation_api_pb2.ManipulationApiRequest(walk_to_object_in_image=walk_to)
                    cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=walk_to_request)
                    g_last_distance_sent = g_forward_distance
                    cv2.destroyWindow(window_name)
                    time.sleep(2)
                    break

            cv2.imshow(window_name, g_image_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow(window_name)
                return

        # --- Dual camera grasp ---
        robot.logger.info("Dual camera preview for grasping. Click on object to pick.")
        dual_window_name = "Front Left & Right Cameras"
        cv2.namedWindow(dual_window_name)
        g_image_click = None
        clicked_camera_index = None

        while True:
            responses = image_client.get_image_from_sources(["frontleft_fisheye_image", "frontright_fisheye_image"])
            if len(responses) != 2:
                continue

            imgs = []
            for resp in responses:
                dtype = np.uint16 if resp.shot.image.pixel_format == 2 else np.uint8
                img = np.frombuffer(resp.shot.image.data, dtype=dtype)
                if resp.shot.image.format == 0:
                    img = img.reshape(resp.shot.image.rows, resp.shot.image.cols)
                else:
                    img = cv2.imdecode(img, -1)
                imgs.append(img)

            combined_img = np.hstack(imgs)
            cv2.setMouseCallback(dual_window_name, dual_mouse_callback, param=imgs)

            if g_cursor_pos is not None and g_cursor_cam is not None:
                cx, cy = g_cursor_pos
                offset_x = 0 if g_cursor_cam == 0 else imgs[0].shape[1]
                cv2.line(combined_img, (offset_x, cy), (offset_x + imgs[g_cursor_cam].shape[1], cy), (0, 255, 0), 1)
                cv2.line(combined_img, (offset_x + cx, 0), (offset_x + cx, combined_img.shape[0]), (0, 255, 0), 1)
                cv2.circle(combined_img, (offset_x + cx, cy), 4, (0, 0, 255), -1)

            if g_image_click is not None and clicked_camera_index is not None:
                cx, cy = g_image_click
                offset_x = 0 if clicked_camera_index == 0 else imgs[0].shape[1]
                cv2.line(combined_img, (offset_x, cy), (offset_x + imgs[clicked_camera_index].shape[1], cy), (255, 0, 0), 2)
                cv2.line(combined_img, (offset_x + cx, 0), (offset_x + cx, combined_img.shape[0]), (255, 0, 0), 2)
                cv2.circle(combined_img, (offset_x + cx, cy), 6, (0, 0, 255), -1)

            cv2.imshow(dual_window_name, combined_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if g_image_click is not None and clicked_camera_index is not None:
                pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
                grasp = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=pick_vec,
                    transforms_snapshot_for_camera=responses[clicked_camera_index].shot.transforms_snapshot,
                    frame_name_image_sensor=responses[clicked_camera_index].shot.frame_name_image_sensor,
                    camera_model=responses[clicked_camera_index].source.pinhole,
                    grasp_params=manipulation_api_pb2.GraspParams(
                        grasp_palm_to_fingertip=0.1,  # how far gripper closes
                    )
                )
                grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
                cmd_response = manipulation_api_client.manipulation_api_command(manipulation_api_request=grasp_request)

                # Wait for grasp to complete
                while True:
                    feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                        manipulation_cmd_id=cmd_response.manipulation_cmd_id
                    )
                    response = manipulation_api_client.manipulation_api_feedback_command(
                        manipulation_api_feedback_request=feedback_request
                    )
                    print(f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}')
                    if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                                                  manipulation_api_pb2.MANIP_STATE_GRASP_FAILED]:
                        break
                    time.sleep(0.25)

                if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                    robot.logger.info('Grasp succeeded, moving arm to carry position for 10 seconds...')
                    carry_cmd = RobotCommandBuilder.arm_carry_command()
                    command_client.robot_command(carry_cmd)
                    time.sleep(10)  # hold carry position for 10 seconds

                    robot.logger.info('Finished grasp sequence. Sitting down and powering off.')
                    blocking_sit(command_client, timeout_sec=10)
                    robot.power_off(cut_immediately=False, timeout_sec=20)
                    assert not robot.is_powered_on(), 'Robot power off failed.'
                    robot.logger.info('Robot safely powered off.')
                    break

def main():
    parser = argparse.ArgumentParser()
    add_base_arguments(parser)
    parser.add_argument("-i", "--image-source", help="Camera source", default="frontleft_fisheye_image")
    options = parser.parse_args()

    try:
        walk_to_object_live(options)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
