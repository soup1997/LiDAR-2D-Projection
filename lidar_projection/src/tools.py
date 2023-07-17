#!/usr/bin/env python3

import rospy

x_min, x_max = 0, 0
y_min, y_max = 0, 0
z_min, z_max = 0, 0

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = rospy.Time.now()
        result = func(*args, **kwargs)
        end_time = rospy.Time.now()

        execution_time = (end_time - start_time).to_sec()
        print(f"Execution time of {func.__name__}:  {execution_time} seconds")

        return result

    return wrapper

def normalize_intensity(intensity):
    intensity /= 100.0
    intensity *= 255
    return intensity

def normalize(x, min=0, max=100):
    x_new = ((x - min) / (max - min)) * 255

    return x_new

def find_minmax(pcd):
    global x_min, x_max, y_min, y_max, z_min, z_max

    x_points = pcd[:, 0]
    y_points = pcd[:, 1]
    z_points = pcd[:, 2]

    x_max = max(x_max, max(x_points))
    y_max = max(y_max, max(y_points))
    z_max = max(z_max, max(z_points))

    x_min = min(x_min, min(x_points))
    y_min = min(y_min, min(y_points))
    z_min = min(z_min, min(z_points))

    return (x_min, y_min, z_min), (x_max, y_max, z_max)