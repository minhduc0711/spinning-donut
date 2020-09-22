import os
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pygame
from pygame import Color

from utils import generate_torus, rotate_about_x_axis, rotate_about_y_axis,\
    rotate_about_z_axis


LUMINANCE = ".,-~:;=!*#$@"

# settings
fps = 30
size_x = 150
size_y = 80
K1 = 150  # screen distance
K2 = 5
r1 = 1
r2 = 2

# num_thetas = int(np.pi * 2 / 0.07)
# num_phis = int(np.pi * 2 / 0.07)
num_thetas = 50
num_phis = 75
angle_x_step = 0.07
angle_z_step = 0.03

angle_x = 1
angle_z = 1

light_direction = (0, 1, -4)
light_direction /= np.linalg.norm(light_direction)

# pygame.init()
# main_surface = pygame.display.set_mode((size_x, size_y))

while True:
    # ev = pygame.event.poll()    # Look for any event
    # if ev.type == pygame.QUIT:  # Window close button clicked?
    #     break

    t0 = time.time()

    # rotate the donut about all 3 axis
    torus_points, torus_normals = generate_torus(num_thetas, num_phis, r1, r2)
    torus_points = rotate_about_x_axis(torus_points, angle_x)
    torus_points = rotate_about_z_axis(torus_points, angle_z)
    torus_normals = rotate_about_x_axis(torus_normals, angle_x)
    torus_normals = rotate_about_z_axis(torus_normals, angle_z)
    angle_x += angle_x_step
    angle_z += angle_z_step

    # compute luminance
    lumi_vals = np.dot(torus_normals.T, light_direction)
    lumi_vals = (lumi_vals + 1) / 2 * (len(LUMINANCE) - 1)
    lumi_vals = np.round(lumi_vals).astype(np.int)

    xs, ys, zs = torus_points
    # move 3D torus far away from z = 0
    zs_shifted = zs + r1 + r2 + 10 + K2
    zs_inverse = 1 / zs_shifted
    # project 3D torus onto 2D space (screen)
    torus_2d = np.array([K1 * xs * zs_inverse,
                         K1 * ys * zs_inverse])
    # translate 2D torus to positive quadrant
    torus_2d += np.array([size_x / 2, size_y / 2 + 10], dtype=np.int).reshape(2, 1)

    # draw
    # main_surface.fill(Color(0, 0, 0))
    # for i, point in enumerate(torus_2d.T.astype(np.int)):
    #     color_white = Color(255, 255, 255)
    #     h, s, v, a = color_white.hsva
    #     color_white.hsva = h, s, lumi_vals[i], a
    #     pygame.draw.circle(main_surface, color_white, point, 2)
    # pygame.display.flip()

    # axis ordering in ndarray is reverted
    s = np.full((size_y, size_x), " ")
    z_buffer = np.zeros((size_y, size_x))
    for i, point in enumerate(torus_2d.T.astype(np.int)):
        if zs_inverse[i] > z_buffer[point[1], point[0]]:
            s[point[1], point[0]] = LUMINANCE[lumi_vals[i]]
            z_buffer[point[1], point[0]] = zs_inverse[i]
    s = "\n".join(["".join(row) for row in s])
    os.system("clear")
    print(s)

    t1 = time.time()
    spare_time = (1 / fps) - (t1 - t0)
    if spare_time > 0:
        time.sleep(spare_time)

# pygame.quit()
