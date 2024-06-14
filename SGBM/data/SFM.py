import numpy as np
import math
import matplotlib.pyplot as plt


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = plt.imread(img_path)
        self.traffic_light = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind=[]
        self.valid=[]


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    normalized = []
    for point in pts:
        normalized.append([(point[0]-pp[0])/focal, (point[1]-pp[1])/focal])
    return np.array(normalized)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnormalized = []
    for point in pts:
        unnormalized.append([point[0]*focal+pp[0], point[1]*focal+pp[1]])
    return np.array(unnormalized)


def decompose(EM):
    r = EM[:3,:3]
    t = EM[:3, 3]
    tz = t[-1]
    foe = [t[0]/t[2], t[1]/t[2]]
    return r, foe, tz
    # extract R, foe and tZ from the Ego Motion


def rotate(pts, R):
    # rotate the points - pts using R
    rotated = []
    for pt in pts:
        prev = np.array([pt[0], pt[1], 1])
        r = R @ prev
        rotated.append([r[0] / r[2], r[1] / r[2]])
    return np.array(rotated)


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    line_m = (foe[1] - p[1]) / (foe[0] - p[0])
    line_n = (p[1]*foe[0] - foe[1]*p[0]) / (foe[0] - p[0])
    distances = []
    for point in norm_pts_rot:
        distance = abs((line_m*point[0]+line_n-point[1]) / (math.sqrt(line_m*line_m+1)))
        distances.append(distance)
    min_ind = distances.index(min(distances))
    return min_ind, norm_pts_rot[min_ind]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    a = ((tZ * (foe[0] - p_rot[0])) / (p_curr[0] - p_rot[0]) + (tZ * (foe[1] - p_rot[1])) / (p_curr[1] - p_rot[1]))/2
    return a