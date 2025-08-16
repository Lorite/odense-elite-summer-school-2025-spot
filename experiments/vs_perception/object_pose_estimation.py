FOCAL_LENGTH = 552.02910121610671
IMAGE_W = 640
IMAGE_H = 480


def get_translation_in_camera_frame(pixel, camera_height):
    x_pixel, y_pixel = pixel
    x_pixel = float(x_pixel - (IMAGE_W / 2)) / FOCAL_LENGTH
    y_pixel = float(y_pixel - (IMAGE_H / 2)) / FOCAL_LENGTH

    return (x_pixel * camera_height, y_pixel*camera_height, camera_height)
