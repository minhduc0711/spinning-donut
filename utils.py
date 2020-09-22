import numpy as np


def rotate_about_x_axis(points, angle):
    rotation_mat = np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)],
    ])
    return rotation_mat @ points


def rotate_about_y_axis(points, angle):
    rotation_mat = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    return rotation_mat @ points


def rotate_about_z_axis(points, angle):
    rotation_mat = np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    return rotation_mat @ points


def generate_torus(num_thetas, num_phis, r1, r2):
    # generate a 2D circle in 3D space
    thetas = np.linspace(0, 2 * np.pi, num=num_thetas)
    circle = np.array(
        [r2 + r1 * np.cos(thetas), r1 * np.sin(thetas), np.zeros(num_thetas)]
    )
    circle_normals = np.array([np.cos(thetas), np.sin(thetas), np.zeros(num_thetas)])

    torus_points = []
    torus_normals = []
    for phi in np.linspace(0, 2 * np.pi, num=num_phis):
        # rotate circle around y-axis to create a torus
        torus_points.append(rotate_about_y_axis(circle, phi))
        # calculate surface normals of torus
        torus_normals.append(rotate_about_y_axis(circle_normals, phi))
    torus_points = np.concatenate(torus_points, axis=1)
    torus_normals = np.concatenate(torus_normals, axis=1)
    return torus_points, torus_normals
