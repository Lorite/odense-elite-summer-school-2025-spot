FOCAL_LENGTH = 552.02910121610671
IMAGE_W = 640
IMAGE_H = 480


def camera_T_obj(object_pixel, camera_height):
    x_pixel, y_pixel = object_pixel
    x_pixel = float(x_pixel - (IMAGE_W / 2)) / FOCAL_LENGTH
    y_pixel = float(y_pixel - (IMAGE_H / 2)) / FOCAL_LENGTH

    return (x_pixel * camera_height, y_pixel*camera_height, camera_height)


def camera_yaw_obj(rect):
    ((center_x, center_y), (width, height), angle) = rect
    yaw = angle
    if width < height:
        yaw += 90
        if yaw > 90:
            yaw -= 180
    yaw *= -1
    return yaw
