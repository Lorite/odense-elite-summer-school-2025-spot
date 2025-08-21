from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectProps:
    hover_offset: float
    grasp_dist: float
    dist_to_floor: float
    gripper_open_fraction: float
    gripper_max_vel: float


OBJECTS: dict[str, ObjectProps] = {
    # Pen
    "pen": ObjectProps(
        hover_offset=0.5,
        grasp_dist=0.2,
        dist_to_floor=0.05,
        gripper_open_fraction=0.05,
        gripper_max_vel=1.0,
    ),
    # Cup variants
    "cup_upright": ObjectProps(
        hover_offset=0.5,
        grasp_dist=0.20,
        dist_to_floor=0.115,
        gripper_open_fraction=0.55,
        gripper_max_vel=0.5,
    ),
    "cup_down": ObjectProps(
        hover_offset=0.5,
        grasp_dist=0.15,
        dist_to_floor=0.115,
        gripper_open_fraction=0.40,
        gripper_max_vel=0.5,
    ),
    "cup_side": ObjectProps(
        hover_offset=0.5,
        grasp_dist=0.20,
        dist_to_floor=0.07,
        gripper_open_fraction=0.50,
        gripper_max_vel=0.5,
    ),
    # Transparent object
    "transparent": ObjectProps(
        hover_offset=0.5,
        grasp_dist=0.10,
        dist_to_floor=0.08,
        gripper_open_fraction=0.25,
        gripper_max_vel=0.25,
    ),
}
