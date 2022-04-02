"""
@author: Vincent Bonnet
@description : core objects not used to describe a scene
"""

import math
import numpy as np
import numba
from jit.maths import normalize
import random

# A per-thread fixed memory pool to prevent memory allocation  and contains
# . pre-allocated arrays
# . pre-allocated ray (origin, direction)
# . pre_allocated hits
@numba.jitclass([('v', numba.float64[:,:]),
                 ('ray_o', numba.float64[:]),
                 ('ray_d', numba.float64[:]),
                 ('depth', numba.int32),
                 ('total_intersection', numba.int64),
                 ('result', numba.float64[:]), # result of trace
                 # hit data are stored for each hit (depth)
                 ('hit_t', numba.float64[:]),  # ray distance as double
                 ('hit_p', numba.float64[:,:]), # hit positon
                 ('hit_in', numba.float64[:,:]), # hit interpolated normal
                 ('hit_n', numba.float64[:,:]), # hit face normal
                 ('hit_tn', numba.float64[:,:]), # hit face tangent
                 ('hit_bn', numba.float64[:,:]), # hit face binormal
                 ('hit_face_id', numba.int32[:]), # hit face id
                 ('hit_material', numba.float64[:,:]), # emittance/reflectance as np.empty(3)
                 ('hit_materialtype', numba.int32[:])]) # material type

class MemoryPool:
    def __init__(self, num_samples):
        self.v = np.empty((3,3)) # pool of vectors
        self.ray_o = np.empty(3) # used for ray origin
        self.ray_d = np.empty(3) # used for ray direction
        self.depth = -1         # depth counter
        self.total_intersection = 0    # total number ray vs element intersection
        self.result = np.empty(3)
        # hit
        self.hit_t = np.empty(num_samples)
        self.hit_p = np.empty((num_samples, 3))
        self.hit_in = np.empty((num_samples, 3)) # interpolated normal
        self.hit_n = np.empty((num_samples, 3)) # face normal
        self.hit_tn = np.empty((num_samples, 3)) # face tangent
        self.hit_bn = np.empty((num_samples, 3)) # face binormal
        self.hit_face_id = np.empty(num_samples, np.int32)
        self.hit_material = np.empty((num_samples, 3))
        self.hit_materialtype = np.empty(num_samples, np.int32)

    def valid_hit(self):
        if self.hit_t[self.depth] >= 0.0:
            return True
        return False

    def next_hit(self):
        self.depth += 1
        self.hit_t[self.depth] = -1 # make the hit invalid

@numba.jitclass([('t', numba.float64), # ray distance as double
                 ('p', numba.float64[:]),
                 ('n', numba.float64[:]), # hit normal as np.empty(3)
                 ('tn', numba.float64[:]), # hit tangent as np.empty(3)
                 ('bn', numba.float64[:]), # hit binormal as np.empty(3)
                 ('face_id', numba.int32), # face id
                 ('reflectance', numba.float64[:]), # reflectance as np.empty(3)
                 ('emittance', numba.float64[:])]) # emittance as np.empty(3)
class Hit:
    def __init__(self):
        self.t = -1.0 # ray distance
        self.face_id = -1

    def valid(self):
        if self.t >= 0.0:
            return True
        return False

@numba.jitclass([('origin', numba.float64[:]),
                 ('width', numba.int32),
                 ('height', numba.int32),
                 ('fovx', numba.float64),
                 ('fovy', numba.float64),
                 ('tan_fovx', numba.float64),
                 ('tan_fovy', numba.float64),
                 ('dir_z', numba.float64),
                 ('supersampling', numba.int32)])
class Camera:
    def __init__(self, width : int, height : int):
        self.origin = np.zeros(3)
        self.fovx = np.pi / 2
        self.dir_z = -1.0
        self.set_resolution(width, height)
        self.set_supersampling(1)

    def set_resolution(self, width : int, height : int):
        self.width = width
        self.height = height
        self.fovy = float(self.height) / float(self.width) * self.fovx
        self.tan_fovx = math.tan(self.fovx*0.5)
        self.tan_fovy = math.tan(self.fovy*0.5)

    def set_supersampling(self, supersampling):
        self.supersampling = supersampling

    def get_ray(self, i : int, j : int, sx : int, sy : int, mempool):
        # i, j : pixel position in the image
        # sx, sy : subpixel location
        # Jitter sampling
        dx = ((random.random() + sx) / self.supersampling) - 0.5
        dy = ((random.random() + sy) / self.supersampling) - 0.5
        x = (2.0 * i - (self.width-1) + dx) / (self.width-1) * self.tan_fovx
        y = (2.0 * j - (self.height-1) + dy) / (self.height-1) * self.tan_fovy
        mempool.ray_o[0] = self.origin[0]
        mempool.ray_o[1] = self.origin[1]
        mempool.ray_o[2] = self.origin[2]
        mempool.ray_d[0] = x
        mempool.ray_d[1] = y
        mempool.ray_d[2] = self.dir_z
        mempool.depth = -1 # no hit
        normalize(mempool.ray_d)
