import numpy as np


def get_x_rotation_mat(angle):
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)],
    ])


def get_y_rotation_mat(angle):
    return np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])


def get_z_rotation_mat(angle):
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


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
        torus_points.append(get_y_rotation_mat(phi) @ circle)
        # calculate surface normals of torus
        torus_normals.append(get_y_rotation_mat(phi) @ circle_normals)
    torus_points = np.concatenate(torus_points, axis=1)
    torus_normals = np.concatenate(torus_normals, axis=1)
    return torus_points, torus_normals
