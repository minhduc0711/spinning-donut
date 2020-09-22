import os
import subprocess
import time

import numpy as np
import numpy_indexed as npi

from utils import generate_torus, get_x_rotation_mat, get_z_rotation_mat


LUMINANCE_CHARS = ".,-~:;=!*#$@"

# settings
fps = 30
# torus radiuses
r1 = 1
r2 = 2
K2 = 5

num_thetas = int(np.pi * 2 / 0.07)
num_phis = int(np.pi * 2 / 0.02)
angle_x_step = 0.07
angle_z_step = 0.03
light_direction = (0, 1, -1)

# prepare for main loop
light_direction /= np.linalg.norm(light_direction)
angle_x = 1
angle_z = 1
init_torus, init_normals = generate_torus(num_thetas, num_phis, r1, r2)
print("\x1b[?25l")

while True:
    try:
        t0 = time.time()
        # scale donut based on terminal size
        size_y, size_x = subprocess.check_output(['stty', 'size']).split()
        size_x, size_y = int(size_x), int(size_y)
        K1 = int(2.38095238 * min(size_x, size_y) + 2.85714286)

        # rotate the donut about all 3 axis
        rotation_mat = get_z_rotation_mat(angle_z) @ get_x_rotation_mat(angle_x)
        torus_points = rotation_mat @ init_torus
        torus_normals = rotation_mat @ init_normals
        angle_x += angle_x_step
        angle_z += angle_z_step

        # compute luminance
        lumi_vals = np.dot(torus_normals.T, light_direction)
        # convert from (-1, 1) to ascii luminance range
        lumi_vals = (lumi_vals + 1) / 2 * (len(LUMINANCE_CHARS))
        lumi_vals = np.floor(lumi_vals).astype(np.int)

        xs, ys, zs = torus_points
        # move 3D torus far away from z = 0
        zs_shifted = zs + r1 + r2 + 10 + K2
        zs_inverse = 1 / zs_shifted
        # project 3D torus onto 2D space (screen)
        torus_2d = np.array([K1 * xs * zs_inverse,
                             K1 * ys * zs_inverse])
        # translate 2D torus to positive quadrant
        torus_2d += np.array([size_x / 2, size_y / 2], dtype=np.int).reshape(2, 1)
        torus_2d = torus_2d.astype(np.int)

        # z-buffer
        coords_2d, lumi_idxs = npi.group_by(torus_2d, axis=1).argmax(zs_inverse)
        # "render"
        s = np.full((size_y, size_x), " ")
        for i in range(lumi_idxs.shape[0]):
            x, y = coords_2d[:, i]
            intensity = lumi_vals[lumi_idxs[i]]
            s[y, x] = LUMINANCE_CHARS[intensity]
        s = "\n".join(["".join(row) for row in s])
        os.system("clear")  # clear the console
        print(s)

        t1 = time.time()
        # fps handling
        spare_time = (1 / fps) - (t1 - t0)
        if spare_time > 0:
            time.sleep(spare_time)
        # print(t1 -t0)
    except KeyboardInterrupt:
        # show terminal cursor again
        print("\x1b[?25h")
        break
