import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk
import threading
import cv2 as cv
import copy
import numpy as np
import time
import math
from multiprocessing import shared_memory
import struct

from spot import Spot, img_spot_to_opencv, calculate_camera_height, translation_camera_to_gpe, translation_gpe_to_flat_body, get_camera_yaw
import object_detection
import object_pose_estimation

W_IMG_0 = 700
W_IMG_1_TO_4 = 300


class GUIInput():
    def __init__(self, img_0, img_1, img_2, img_3, img_4, rect, pose, fps):
        self.img_0 = img_0
        self.img_1 = img_1
        self.img_2 = img_2
        self.img_3 = img_3
        self.img_4 = img_4
        self.rect = rect
        self.pose = pose
        self.fps = fps


gui_input = GUIInput(None, None, None, None, None, None, None, None)
gui_input_lock = threading.Lock()


class GUIOutput():
    def __init__(self, img_blur, color_threshold, mask_opening_radius):
        self.img_blur = img_blur
        self.color_threshold = color_threshold
        self.mask_opening_radius = mask_opening_radius


gui_output = GUIOutput(1, 3.0, 1)
gui_output_lock = threading.Lock()


def resize_image(img, max_width):
    h, w = img.shape[:2]
    ratio = max_width / w
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
    return resized


def cv_to_pil_img(cv_img):
    img_rgb = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def get_bgr_img_from_mask(mask):
    """
    Gets BGR image from bool mask.

    Args:
        mask: Boolean mask.

    Returns:
        The BGR image.
    """
    img = (mask.astype(np.uint8)) * 255
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


def draw_line(img, center, angle, color, thickness):
    """
    Draws a line across the entire image through center point at the specified angle.

    Args:
        img: The image to draw on.
        center: Center point the line should pass through.
        angle: Angle of the line.
        color: Color of the line.
        thickness: Thickness of the line.
    """
    theta = np.deg2rad(angle)

    img_h, img_w = img.shape[:2]
    length = int(np.sqrt(np.pow(img_w, 2) + np.pow(img_h, 2)))

    dx = np.cos(theta)
    dy = np.sin(theta)

    x1 = int(center[0] - dx * length)
    y1 = int(center[1] - dy * length)
    x2 = int(center[0] + dx * length)
    y2 = int(center[1] + dy * length)

    cv.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rect(img, rect, color=(0, 0, 255), thickness=1):
    """
    Draws a rectangle with orientation lines and center point on the image.

    Args:
        img: The image to draw on.
        rect: The rectangle. May be None.
        color: Color of the line.
        thickness: Thickness of the line.
    """
    if rect is None:
        return

    box = cv.boxPoints(rect)
    box = np.intp(box)
    cv.drawContours(img, [box], 0, color, thickness)

    center = tuple(map(int, rect[0]))
    angle = rect[2]
    draw_line(img, center, angle, color, thickness)
    draw_line(img, center, angle+90, color, thickness)


root = tk.Tk()
root.title("Perception")

default_font = font.nametofont("TkDefaultFont")
default_font.configure(size=14)

top_frame = tk.Frame(root)
top_frame.pack(fill="x", anchor="w")

img_0_lbl = tk.Label(top_frame)
img_0_lbl.grid(row=0, column=0, sticky="w")

data_frame = tk.Frame(top_frame)
data_frame.grid(row=0, column=1, sticky="nw")

tk.Label(data_frame, text="Position").grid(row=0, column=0, sticky="w")
tk.Label(data_frame, text="FPS").grid(row=4, column=0, sticky="w")
tk.Label(data_frame, text="img_blur").grid(row=6, column=0, sticky="w")
tk.Label(data_frame, text="color_threshold").grid(row=7, column=0, sticky="w")
tk.Label(data_frame, text="mask_opening_radius").grid(
    row=8, column=0, sticky="w")

txt_pos_x_lbl = tk.Label(data_frame)
txt_pos_x_lbl.grid(row=0, column=1, sticky="w")

txt_pos_y_lbl = tk.Label(data_frame)
txt_pos_y_lbl.grid(row=1, column=1, sticky="w")

txt_pos_z_lbl = tk.Label(data_frame)
txt_pos_z_lbl.grid(row=2, column=1, sticky="w")

txt_fps_lbl = tk.Label(data_frame)
txt_fps_lbl.grid(row=4, column=1, sticky="w")

entry_1 = tk.Entry(data_frame, width=6)
entry_1.insert(0, str(gui_output.img_blur))
entry_1.grid(row=6, column=1)
entry_1.bind("<Return>", lambda e: value_changed())
entry_1.bind("<FocusOut>", lambda e: value_changed())
entry_1.bind("<FocusIn>", lambda e: select_all(e))

entry_2 = tk.Entry(data_frame, width=6)
entry_2.insert(0, str(gui_output.color_threshold))
entry_2.grid(row=7, column=1)
entry_2.bind("<Return>", lambda e: value_changed())
entry_2.bind("<FocusOut>", lambda e: value_changed())
entry_2.bind("<FocusIn>", lambda e: select_all(e))

entry_3 = tk.Entry(data_frame, width=6)
entry_3.insert(0, str(gui_output.mask_opening_radius))
entry_3.grid(row=8, column=1)
entry_3.bind("<Return>", lambda e: value_changed())
entry_3.bind("<FocusOut>", lambda e: value_changed())
entry_3.bind("<FocusIn>", lambda e: select_all(e))

data_frame.grid_rowconfigure(3, minsize=10)
data_frame.grid_rowconfigure(5, minsize=20)

bottom_frame = tk.Frame(root)
bottom_frame.pack()

img_1_lbl = tk.Label(bottom_frame)
img_1_lbl.grid(row=0, column=0)

img_2_lbl = tk.Label(bottom_frame)
img_2_lbl.grid(row=0, column=1)

img_3_lbl = tk.Label(bottom_frame)
img_3_lbl.grid(row=0, column=2)

img_4_lbl = tk.Label(bottom_frame)
img_4_lbl.grid(row=0, column=3)


def update_gui():
    with gui_input_lock:
        input = copy.deepcopy(gui_input)

    if input.img_0 is not None:
        img = input.img_0
        if input.rect is not None:
            draw_rect(img, input.rect)
        img = resize_image(img, W_IMG_0)
        tk_img = ImageTk.PhotoImage(cv_to_pil_img(img))
        img_0_lbl.config(image=tk_img)
        img_0_lbl.image = tk_img

    bottom_inuput_imgs = [input.img_1, input.img_2,
                          input.img_3, input.img_4]
    bottom_img_lbls = [img_1_lbl, img_2_lbl, img_3_lbl, img_4_lbl]
    for i in range(4):
        if bottom_inuput_imgs[i] is not None:
            img = bottom_inuput_imgs[i]
            if i >= 1:
                img = get_bgr_img_from_mask(img)
            if input.rect is not None:
                draw_rect(img, input.rect)
            img = resize_image(img, W_IMG_1_TO_4)
            tk_img = ImageTk.PhotoImage(cv_to_pil_img(img))
            bottom_img_lbls[i].config(image=tk_img)
            bottom_img_lbls[i].image = tk_img

    txt_pos_x_lbl.config(
        text=f"{input.pose[0]:.2f}" if input.pose is not None else "")
    txt_pos_y_lbl.config(
        text=f"{input.pose[1]:.2f}" if input.pose is not None else "")
    txt_pos_z_lbl.config(
        text=f"{input.pose[2]:.2f}" if input.pose is not None else "")
    txt_fps_lbl.config(
        text=f"{input.fps:.1f}" if input.fps is not None else "")

    root.after(60, update_gui)


def select_all(event):
    event.widget.select_range(0, tk.END)
    event.widget.icursor(tk.END)
    return "break"


def value_changed():
    global gui_output

    try:
        val1 = int(entry_1.get())
        if val1 <= 0:
            raise ValueError
    except ValueError:
        entry_1.delete(0, "end")
        entry_1.insert(0, str(gui_output.img_blur))
        return

    try:
        val2 = float(entry_2.get())
        if val2 <= 0:
            raise ValueError
    except ValueError:
        entry_2.delete(0, "end")
        entry_2.insert(0, str(gui_output.color_threshold))
        return
    if '.' not in entry_2.get():
        entry_2.insert("end", ".")
    if entry_2.get()[-1] == '.':
        entry_2.insert("end", "0")

    try:
        val3 = int(entry_3.get())
        if val3 <= 0:
            raise ValueError
    except ValueError:
        entry_3.delete(0, "end")
        entry_3.insert(0, str(gui_output.mask_opening_radius))
        return

    with gui_output_lock:
        gui_output = GUIOutput(val1, val2, val3)


update_gui()


def perception():
    global gui_input

    shm = shared_memory.SharedMemory(
        create=True, size=32, name='shm_goal')
    buffer = shm.buf

    try:

        spot = Spot()

        t = time.time()
        while True:
            img_spot = spot.get_spot_img()
            img = img_spot_to_opencv(img_spot)

            robot_state_transforms_snapshot = spot.get_robot_state_transforms_snapshot()
            img_transforms_snapshot = img_spot.shot.transforms_snapshot

            camera_height = calculate_camera_height(
                robot_state_transforms_snapshot, img_transforms_snapshot)

            with gui_output_lock:
                gui_output_copy = copy.deepcopy(gui_output)

            detect_object_result = object_detection.detect_object(img,
                                                                  gui_output_copy.img_blur,
                                                                  gui_output_copy.color_threshold,
                                                                  gui_output_copy.mask_opening_radius)
            if detect_object_result.rect is not None:
                translation_camera = object_pose_estimation.get_translation_in_camera_frame(
                    detect_object_result.rect[0], camera_height)

                translation_gpe = (translation_camera_to_gpe(
                    translation_camera, robot_state_transforms_snapshot, img_transforms_snapshot))

                translation_gpe = (translation_gpe[0], translation_gpe[1], 0)

                translation_flat_body = translation_gpe_to_flat_body(
                    translation_gpe, robot_state_transforms_snapshot)

                camera_yaw = math.degrees(get_camera_yaw(
                    robot_state_transforms_snapshot, img_transforms_snapshot))

                object_yaw = detect_object_result.rect[2]
                if detect_object_result.rect[1][0] < detect_object_result.rect[1][1]:
                    object_yaw += 90
                    if object_yaw > 90:
                        object_yaw -= 180
                object_yaw *= -1

                yaw_goal = camera_yaw + object_yaw

                buffer[:32] = struct.pack(
                    'dddd', translation_flat_body[0], translation_flat_body[1], translation_flat_body[2], yaw_goal)

            else:
                translation_camera = None

            t_now = time.time()
            fps = 1.0 / (t_now - t)
            t = t_now

            with gui_input_lock:
                gui_input = GUIInput(img,
                                     detect_object_result.img_pp,
                                     detect_object_result.color_mask,
                                     detect_object_result.mask_pp,
                                     detect_object_result.largest_component_mask,
                                     detect_object_result.rect,
                                     translation_camera,
                                     fps)
    finally:
        shm.close()
        shm.unlink()


thread = threading.Thread(target=perception, daemon=True)
thread.start()


root.mainloop()
