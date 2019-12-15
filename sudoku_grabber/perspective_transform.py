import cv2
import numpy as np
from numpy.linalg import pinv


def calculate_distance(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def calculate_max_dimensions(corners):
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    width1 = calculate_distance(br, bl)
    width2 = calculate_distance(tr, tl)
    max_width = max(int(width1), int(width2))

    height1 = calculate_distance(tr, br)
    height2 = calculate_distance(tl, bl)
    max_height = max(int(height1), int(height2))

    return max_width, max_height


def calculate_transformation_matrix(corners, target_corners):
    x = np.array([p[0] for p in corners])
    y = np.array([p[1] for p in corners])
    xp = np.array([p[0] for p in target_corners])
    yp = np.array([p[1] for p in target_corners])

    A = np.empty((0, 8))
    for (xo, yo, xn, yn) in zip(x, y, xp, yp):
        row = np.array([xo, yo, 1, 0, 0, 0, -xn * xo, -xn * yo]).reshape(1, -1)
        A = np.vstack((A, row))
    for (xo, yo, xn, yn) in zip(x, y, xp, yp):
        row = np.array([0, 0, 0, xo, yo, 1, -yn * xo, -yn * yo]).reshape(1, -1)
        A = np.vstack((A, row))

    B = [xp[0], xp[1], xp[2], xp[3], yp[0], yp[1], yp[2], yp[3]]
    B = np.array(B).reshape(-1, 1)

    m = pinv(A.T @ A) @ A.T @ B

    return m


def transform(m, points):
    m = np.vstack((m, np.ones((1, 1))))
    m = m.reshape((3, 3))

    x = np.array([p[0] for p in points]).reshape(1, -1)
    y = np.array([p[1] for p in points]).reshape(1, -1)

    v = np.empty((0, x.shape[1]))
    v = np.vstack((v, x))
    v = np.vstack((v, y))
    v = np.vstack((v, np.ones((1, x.shape[1]))))

    r = (m @ v) / (m[-1, :] @ v)

    result = []
    for row in r.T:
        result.append((row[0], row[1]))

    return result


def update_map(map_x, map_y, m):
    points = []
    for i in range(map_x.shape[0]):
        for j in range(map_y.shape[1]):
            points.append((j, i))

    new_coordinates = transform(m, points)
    counter = 0
    for i in range(map_x.shape[0]):
        for j in range(map_y.shape[1]):
            x, y = new_coordinates[counter]
            map_x[i, j] = x
            map_y[i, j] = y
            counter += 1


def crop_and_warp(input_image, corner_points):
    image = input_image.copy()

    w, h = calculate_max_dimensions(corner_points)
    target_corners = [
        (0, 0),
        (w, 0),
        (w, h),
        (0, h)
    ]
    transformation_matrix = calculate_transformation_matrix(target_corners, corner_points)

    map_x = np.zeros_like(image, dtype=np.float32)
    map_y = np.zeros_like(image, dtype=np.float32)
    update_map(map_x, map_y, transformation_matrix)
    remapped_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    remapped_image = remapped_image[0:h, 0:w]

    return remapped_image
