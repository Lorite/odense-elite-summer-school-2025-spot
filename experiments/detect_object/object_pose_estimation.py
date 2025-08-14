import os

import bosdyn.client
from bosdyn.client.robot_state import RobotStateClient, robot_state_pb2
from bosdyn.client.image import ImageClient


def get_camera_height(robot_state_transforms_snapshot, img_transforms_snapshot):
    gpe_tform_body = bosdyn.client.frame_helpers.get_a_tform_b(
        robot_state_transforms_snapshot, 'gpe', 'body')
    body_tform_camera = bosdyn.client.frame_helpers.get_a_tform_b(
        img_transforms_snapshot, 'body', 'hand_color_image_sensor')
    gpe_tform_camera = gpe_tform_body * body_tform_camera
    return gpe_tform_camera.z


if __name__ == '__main__':
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ROBOT_IP = os.getenv("ROBOT_IP")

    print("Establishing connection to robot...")
    sdk = bosdyn.client.create_standard_sdk('detect_object')
    robot = sdk.create_robot(ROBOT_IP)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    print("Connection established!")

    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    while True:

        robot_state_transforms_snapshot = robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot

        img_transforms_snapshot = image_client.get_image_from_sources(['hand_color_image'])[
            0].shot.transforms_snapshot

        height = get_camera_height(
            robot_state_transforms_snapshot, img_transforms_snapshot)

        print(height)
