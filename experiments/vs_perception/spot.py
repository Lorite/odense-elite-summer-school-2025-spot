import os
import math
import cv2 as cv
import numpy as np

import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient, build_image_request


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROBOT_IP = os.getenv("ROBOT_IP")


def img_spot_to_opencv(img_spot):
    '''
    Gets image usable within OpenCV from an image obtained from spot.

    Args:
        img: Image obtaines by spot (JPEG or RAW RGB_U8 image)

    Returns:
        Image usable within OpenCV
    '''
    img_opencv = np.frombuffer(img_spot.shot.image.data, dtype=np.uint8)
    if img_spot.shot.image.format == image_pb2.Image.FORMAT_JPEG:
        img_opencv = cv.imdecode(img_opencv, -1)
    elif img_spot.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img_opencv = img_opencv.reshape(
            (img_spot.shot.image.rows, img_spot.shot.image.cols, 3))
    return img_opencv


def gpe_tform_camera(robot_state_transforms_snapshot, img_transforms_snapshot):
    gpe_tform_body = bosdyn.client.frame_helpers.get_a_tform_b(
        robot_state_transforms_snapshot, 'gpe', 'body')
    body_tform_camera = bosdyn.client.frame_helpers.get_a_tform_b(
        img_transforms_snapshot, 'body', 'hand_color_image_sensor')
    gpe_tform_camera = gpe_tform_body * body_tform_camera
    return gpe_tform_camera


def camera_height(gpe_tform_camera):
    return gpe_tform_camera.z


def flat_body_tform_gpe(robot_state_transforms_snapshot):
    flat_body_tform_gpe = bosdyn.client.frame_helpers.get_a_tform_b(
        robot_state_transforms_snapshot, 'flat_body', 'gpe')
    return flat_body_tform_gpe


def flat_body_tform_camera(robot_state_transforms_snapshot, img_transforms_snapshot):
    flat_body_T_body = bosdyn.client.frame_helpers.get_a_tform_b(
        robot_state_transforms_snapshot, "flat_body", "body")
    body_T_camera = bosdyn.client.frame_helpers.get_a_tform_b(
        img_transforms_snapshot, "body", "hand_color_image_sensor")
    flat_body_T_camera = flat_body_T_body * body_T_camera
    return flat_body_T_camera


def flat_body_yaw_camera(robot_state_transforms_snapshot, img_transforms_snapshot):
    tform = flat_body_tform_camera(
        robot_state_transforms_snapshot, img_transforms_snapshot)
    rot = tform.rotation.to_matrix()
    x_axis = rot[:, 0]
    x_proj = np.array([x_axis[0], x_axis[1]])
    yaw = np.arctan2(x_proj[1], x_proj[0])
    yaw += math.pi / 2
    yaw = math.degrees(yaw)
    return yaw


class Spot():

    def __init__(self):
        print("Establishing connection to spot...")
        sdk = bosdyn.client.create_standard_sdk('vs_perception')
        robot = sdk.create_robot(ROBOT_IP)
        bosdyn.client.util.authenticate(robot)
        robot.sync_with_directory()
        robot.time_sync.wait_for_sync()
        self.image_client = robot.ensure_client(
            ImageClient.default_service_name)
        self.robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name)
        print("Connection to spot established!")

    def get_img_spot(self):
        spot_img_request = build_image_request('hand_color_image',
                                               image_format=image_pb2.Image.FORMAT_JPEG,
                                               pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                                               quality_percent=100)
        spot_img = self.image_client.get_image([spot_img_request])[0]
        return spot_img

    def get_robot_state_transforms_snapshot(self):
        return self.robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot
