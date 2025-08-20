import time
import struct
from multiprocessing import shared_memory

import spot
import object_detection
import object_pose_estimation


SHM_NAME = 'shm_vs'


class FPSMeasurement():
    def __init__(self):
        self.t = time.time()
        self.fps = None

    def frame(self):
        t_new = time.time()
        t_diff = t_new - self.t
        self.fps = 1.0 / t_diff
        self.t = t_new

    def get_fps(self):
        return self.fps


class PerceptionParameters():
    def __init__(self, img_blur, color_threshold, mask_opening_radius):
        self.img_blur = img_blur
        self.color_threshold = color_threshold
        self.mask_opening_radius = mask_opening_radius


class PerceptionOutput():
    def __init__(self, img, img_pp, color_mask, mask_pp, largest_component_mask, rect, camera_T_obj, camera_yaw_obj, fps):
        self.img = img
        self.img_pp = img_pp
        self.color_mask = color_mask
        self.mask_pp = mask_pp
        self.largest_component_mask = largest_component_mask
        self.rect = rect
        self.camera_T_obj = camera_T_obj
        self.camera_yaw_obj = camera_yaw_obj
        self.fps = fps


def perception(stop_event, get_params_callback, update_out_callback):

    shm = shared_memory.SharedMemory(
        create=True, size=struct.calcsize('dddd'), name=SHM_NAME)
    buffer = shm.buf

    try:
        robot = spot.Spot()

        fps = FPSMeasurement()
        while not stop_event.is_set():

            img_spot = robot.get_img_spot()
            img = spot.img_spot_to_opencv(img_spot)

            robot_state_transforms_snapshot = robot.get_robot_state_transforms_snapshot()
            img_transforms_snapshot = img_spot.shot.transforms_snapshot
            gpe_tform_camera = spot.gpe_tform_camera(
                robot_state_transforms_snapshot, img_transforms_snapshot)
            camera_height = spot.camera_height(gpe_tform_camera)

            params = get_params_callback()

            detect_object_result = object_detection.detect_object(
                img, params.img_blur, params.color_threshold, params.mask_opening_radius)

            if detect_object_result.rect is not None:

                camera_T_obj = object_pose_estimation.camera_T_obj(
                    detect_object_result.rect[0], camera_height)
                gpe_T_obj = gpe_tform_camera.transform_point(*camera_T_obj)
                gpe_T_obj = (gpe_T_obj[0], gpe_T_obj[1], 0)
                flat_body_T_obj = spot.flat_body_tform_gpe(
                    robot_state_transforms_snapshot).transform_point(*gpe_T_obj)

                flat_body_yaw_camera = spot.flat_body_yaw_camera(
                    robot_state_transforms_snapshot, img_transforms_snapshot)
                camera_yaw_obj = object_pose_estimation.camera_yaw_obj(
                    detect_object_result.rect)
                flat_body_yaw_obj = flat_body_yaw_camera + camera_yaw_obj

                buffer[:32] = struct.pack(
                    'dddd', *flat_body_T_obj, flat_body_yaw_obj)

            else:
                camera_T_obj = None
                camera_yaw_obj = None

            fps.frame()
            output = PerceptionOutput(img,
                                      detect_object_result.img_pp,
                                      detect_object_result.color_mask,
                                      detect_object_result.mask_pp,
                                      detect_object_result.largest_component_mask,
                                      detect_object_result.rect,
                                      camera_T_obj,
                                      camera_yaw_obj,
                                      fps.get_fps())
            update_out_callback(output)

    finally:
        shm.close()
        shm.unlink()
