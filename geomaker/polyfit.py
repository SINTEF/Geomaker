from functools import partial
from itertools import combinations
import numpy as np
import scipy.optimize as opt


def filter_none(objs):
    for obj in objs:
        if obj is not None:
            yield obj


def intersect_single(left, right, x, bias='down'):
    if left[0] == x == right[0]:
        if bias == 'down':
            return min(right[1], left[1])
        return max(right[1], left[1])
    if left[0] > right[0]:
        left, right = right, left
    if left[0] <= x <= right[0]:
        y = (x - left[0]) / (right[0] - left[0])
        return (1 - y) * left[1] + y * right[1]
    return None


def intersect_all(x, edges, bias='down'):
    if bias == 'down':
        return min(filter_none(intersect_single(left, right, x, 'down') for (left, right) in edges))
    return max(filter_none(intersect_single(left, right, x, 'up') for (left, right) in edges))


def rectangle_size(edges, l, r):
    lu = intersect_all(l, edges, 'up')
    ld = intersect_all(l, edges, 'down')
    ru = intersect_all(r, edges, 'up')
    rd = intersect_all(r, edges, 'down')

    u = min(lu, ru)
    d = max(ld, rd)
    return (r - l) * (u - d)


def largest_aligned_rectangle(points):
    edges = [(a,b) for a, b in zip(points, points[1:])]
    L = min(pt[0] for pt in points)
    R = max(pt[0] for pt in points)
    def objfun(x):
        l = max(min(x), L)
        r = min(max(x), R)
        return -rectangle_size(edges, l, r)
    result = opt.dual_annealing(objfun, [(L, R), (L, R)])
    l = max(min(result.x), L)
    r = min(max(result.x), R)
    d = max(intersect_all(l, edges, 'down'), intersect_all(r, edges, 'down'))
    u = min(intersect_all(l, edges, 'up'), intersect_all(r, edges, 'up'))
    pts = [np.array([l, d]), np.array([r, d]), np.array([r, u]), np.array([l, u]), np.array([l, d])]
    area = (r - l) * (u - d)
    return pts, area


def largest_rotated_rectangle(points, theta):
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotpts = [rotmat.dot(pt) for pt in points]
    rectpts, area = largest_aligned_rectangle(rotpts)
    rectpts = [rotmat.T.dot(pt) for pt in rectpts]
    return rectpts, area


def largest_rectangle(points):
    def objfun(theta):
        _, area = largest_rotated_rectangle(points, float(theta))
        return -area
    bounds = (-np.pi/4, np.pi/4)
    result = opt.minimize_scalar(objfun, bounds=bounds, method='Bounded')
    pts, area = largest_rotated_rectangle(points, float(result.x))
    return pts, area, result.x


def smallest_aligned_rectangle(points):
    l = min(pt[0] for pt in points)
    r = max(pt[0] for pt in points)
    d = min(pt[1] for pt in points)
    u = max(pt[1] for pt in points)
    pts = [np.array([l, d]), np.array([r, d]), np.array([r, u]), np.array([l, u]), np.array([l, d])]
    area = (r - l) * (u - d)
    return pts, area


def smallest_rotated_rectangle(points, theta):
    rotmat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotpts = [rotmat.dot(pt) for pt in points]
    rectpts, area = smallest_aligned_rectangle(rotpts)
    rectpts = [rotmat.T.dot(pt) for pt in rectpts]
    return rectpts, area


def smallest_rectangle(points):
    def objfun(theta):
        _, area = smallest_rotated_rectangle(points, float(theta))
        return area
    bounds = (-np.pi/4, np.pi/4)
    result = opt.minimize_scalar(objfun, bounds=bounds, method='Bounded')
    pts, area = smallest_rotated_rectangle(points, float(result.x))
    return pts, area, result.x


def polygon_area(corners):
    corners = corners[:-1]
    n = len(corners)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area
