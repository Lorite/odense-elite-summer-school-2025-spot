import os
import tempfile
import struct
import mmap

from . import util
from . import spot
from . import object_detection
from . import object_pose_estimation


ROLLING_MEDIAN_NUM_VALUES = 5


MMAP_FILENAME = os.path.join(tempfile.gettempdir(), "mmap_vs.bin")
MMAP_SIZE = 32


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

    with open(MMAP_FILENAME, 'wb') as f:
        f.write(b"\x00" * MMAP_SIZE)
    with open(MMAP_FILENAME, 'r+b') as f:
        shm = mmap.mmap(f.fileno(), MMAP_SIZE, access=mmap.ACCESS_WRITE)

        try:
            robot = spot.Spot()

            fps = util.FPSMeasurement()
            flat_body_T_obj_median = util.RunningMedianTuple(
                ROLLING_MEDIAN_NUM_VALUES, 3)
            flat_body_yaw_obj_median = util.RunningMedian(
                ROLLING_MEDIAN_NUM_VALUES)

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

                    flat_body_T_obj_median.add_value(flat_body_T_obj)
                    flat_body_yaw_obj_median.add_value(flat_body_yaw_obj)

                    shm[:MMAP_SIZE] = struct.pack(
                        'dddd', *flat_body_T_obj_median.get(), flat_body_yaw_obj_median.get())

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
            os.remove(MMAP_FILENAME)
