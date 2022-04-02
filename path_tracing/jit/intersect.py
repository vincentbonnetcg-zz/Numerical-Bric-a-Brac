"""
@author: Vincent Bonnet
@description : intersection routines
"""

import math
import numba
import numpy as np
from .maths import dot, isclose, triple_product
from .maths import asub

@numba.njit(inline='always')
def ray_aabb(mempool, tv):
    # TODO  - replace tv with bounds
    # bounds[0] # min bounds
    # bounds[1] # max bound
     # min bounds
    min_x = min(min(tv[0][0], tv[1][0]), tv[2][0])
    min_y = min(min(tv[0][1], tv[1][1]), tv[2][1])
    min_z = min(min(tv[0][2], tv[1][2]), tv[2][2])
    # max bounds
    max_x = max(max(tv[0][0], tv[1][0]), tv[2][0])
    max_y = max(max(tv[0][1], tv[1][1]), tv[2][1])
    max_z = max(max(tv[0][2], tv[1][2]), tv[2][2])

    # test if ray starts inside
    if mempool.ray_o[0] >= min_x or mempool.ray_o[0] <= max_x:
        return True

    if mempool.ray_o[1] >= min_y or mempool.ray_o[1] <= max_y:
        return True

    if mempool.ray_o[2] >= min_z or mempool.ray_o[2] <= max_z:
        return True

    # compute t for each intersection plane
    inv_dx = 1.0 / mempool.ray_d[0]
    inv_dy = 1.0 / mempool.ray_d[1]
    inv_dz = 1.0 / mempool.ray_d[2]

    min_tx = (min_x - mempool.ray_o[0]) * inv_dx
    max_tx = (max_x - mempool.ray_o[0]) * inv_dx

    if (min_tx > max_tx): # make sure min is actually the minimum value
        min_tx, max_tx = max_tx, min_tx

    min_ty = (min_y - mempool.ray_o[1]) * inv_dy
    max_ty = (max_y - mempool.ray_o[1]) * inv_dy

    if (min_ty > max_ty):  # make sure min is actually the minimum value
        min_ty, max_ty = max_ty, min_ty

    if ((min_tx > max_ty) or (min_ty > max_tx)):
        return False

    # update min_ty and max_tx
    if (min_ty > min_tx):
        min_tx = min_ty

    if (max_ty < max_tx):
        max_tx = max_ty

    min_tz = (min_z - mempool.ray_o[2]) * inv_dz
    max_tz = (max_z - mempool.ray_o[2]) * inv_dz

    if (min_tz > max_tz):  # make sure min is actually the minimum value
        min_tz, max_tz = max_tz, min_tz

    if ((min_tx > max_tz) or (min_tz > max_tx)):
        return False

    if (min_tz > min_tx):
        min_tx = min_tz

    return True

@numba.njit(inline='always')
def ray_triangle(mempool, tv):
    # Moller-Trumbore intersection algorithm
    asub(tv[1], tv[0], mempool.v[0]) # e1
    asub(tv[2], tv[0], mempool.v[1]) # e2
    asub(mempool.ray_o, tv[0], mempool.v[2]) # ed

    # explicit linear system (Ax=b) for debugging
    #e1 = tv[1] - tv[0]
    #e2 = tv[2] - tv[0]
    #ed = ray_o - tv[0]
    #x = [t, u, v]
    #b = ray_o - tv[0]
    #A = np.zeros((3, 3), dtype=float)
    #A[:,0] = -ray_d
    #A[:,1] = e1
    #A[:,2] = e2
    # solve the system with Cramer's rule
    # det(A) = dot(-ray_d, cross(e1,e2)) = tripleProduct(-ray_d, e1, e2)
    # also det(A) = tripleProduct(ray_d, e1, e2) = -tripleProduct(-ray_d, e1, e2)
    detA = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return 0.0,0.0,-1.0

    invDetA = 1.0 / detA

    u = -triple_product(mempool.ray_d, mempool.v[2], mempool.v[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return 0.0,0.0,-1.0

    v = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[2]) * invDetA
    if (v < 0.0 or u + v > 1.0):
        return 0.0,0.0,-1.0

    t = triple_product(mempool.v[2], mempool.v[0], mempool.v[1]) * invDetA
    return u, v, t

@numba.njit(inline='always')
def ray_quad(mempool, tv):
    # Moller-Trumbore intersection algorithm
    # same than ray_triangle but different condition on v
    asub(tv[1], tv[0], mempool.v[0]) # e1
    asub(tv[2], tv[0], mempool.v[1]) # e2
    asub(mempool.ray_o, tv[0], mempool.v[2]) # ed

    detA = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return 0.0,0.0,-1.0

    invDetA = 1.0 / detA

    u = -triple_product(mempool.ray_d, mempool.v[2], mempool.v[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return 0.0,0.0,-1.0

    v = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[2]) * invDetA
    if (v < 0.0 or v > 1.0):
        return 0.0,0.0,-1.0

    t = triple_product(mempool.v[2], mempool.v[0], mempool.v[1]) * invDetA
    return u, v, t

@numba.njit(inline='always')
def ray_sphere(mempool, sphere_c, sphere_r):
    o = mempool.ray_o - sphere_c
    a = dot(mempool.ray_d, mempool.ray_d)
    b = dot(mempool.ray_d, o) * 2.0
    c = dot(o, o) - sphere_r**2
    # solve ax**2 + bx + c = 0
    dis = b**2 - 4*a*c  # discriminant

    if dis < 0.0:
        # no solution
        return 0.0,0.0,-1.0

    if isclose(dis, 0.0):
        # one solution
        return 0.0, 0.0, -b / 2 * a

    # two solution
    sq = math.sqrt(dis)
    s1 = (-b-sq) / 2*a  # first solution
    s2 = (-b+sq) / 2*a # second solution

    if s1 < 0.0 and s2 < 0.0:
        return 0.0,0.0,-1.0

    t = s2
    if s1 > 0.0 and s2 > 0.0:
        t = np.minimum(s1, s2)
    elif s1 > 0.0:
        t = s1

    return 0.0,0.0, t