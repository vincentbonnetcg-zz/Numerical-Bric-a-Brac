"""
@author: Vincent Bonnet
@description : basic render routines
"""

import time
import math
import random
import numba
import numpy as np
from . import core as jit_core
from .maths import dot, copy, axpy, gamma_correction, clamp, tri_interpolation, normalize
from . import intersect

# pathtracer settings
BLACK = np.zeros(3)
WHITE = np.ones(3)
MAX_DEPTH = 1 # max hit
NUM_SAMPLES = 1 # number of sample per pixel
RANDOM_SEED = 10
INV_PDF = 2.0 * math.pi; # inverse of probability density function
INV_PI = 1.0 / math.pi
SUPERSAMPLING = 2 # supersampling 2x2
CPU_COUNT = 4 # number of cpu
LIGHT_MATERIAL_ID = 1

@numba.njit(inline='always')
def update_ray_from_uniform_distribution(mempool):
    i = mempool.depth
    copy(mempool.ray_o, mempool.hit_p[i])
    # Find ray direction from uniform around hemisphere
    # Unit hemisphere from spherical coordinates
    # the unit  hemisphere is at origin and y is the up vector
    # theta [0, 2*PI) and phi [0, PI/2]
    # px = cos(theta)*sin(phi)
    # py = sin(theta)*sin(phi)
    # pz = cos(phi)
    # A uniform distribution (avoid more samples at the pole)
    # theta = 2*PI*rand()
    # phi = acos(rand())  not phi = PI/2*rand() !
    # Optimization
    # cos(phi) = cos(acos(rand())) = rand()
    # sin(phi) = sin(acos(rand())) = sqrt(1 - rand()^2)
    theta = 2*math.pi*random.random()
    cos_phi = random.random()
    sin_phi = math.sqrt(1.0 - cos_phi**2)
    v0 = math.cos(theta)*sin_phi
    v1 = cos_phi
    v2 = math.sin(theta)*sin_phi
    # compute the world sample
    mempool.ray_d[0] = v0*mempool.hit_bn[i][0] + v1*mempool.hit_n[i][0] + v2*mempool.hit_tn[i][0]
    mempool.ray_d[1] = v0*mempool.hit_bn[i][1] + v1*mempool.hit_n[i][1] + v2*mempool.hit_tn[i][1]
    mempool.ray_d[2] = v0*mempool.hit_bn[i][2] + v1*mempool.hit_n[i][2] + v2*mempool.hit_tn[i][2]

@numba.njit
def ray_tri_details(details, mempool):
    # details from Scene.tri_details()
    skip_face_id = -1
    if mempool.depth >= 0: # skip face based on previous hit
        skip_face_id = mempool.hit_face_id[mempool.depth]
    mempool.next_hit() # use the next allocated hit
    nearest_t = np.finfo(numba.float64).max
    nearest_u = 0.0
    nearest_v = 0.0
    data = details[0]
    tri_vertices = data.tri_vertices
    hit_id = -1
    # intersection test with triangles
    num_triangles = len(tri_vertices)
    for i in range(num_triangles):
        if i == skip_face_id:
            continue
        #if intersect.ray_aabb(mempool, tri_vertices[i])==False:
        #    continue
        uvt = intersect.ray_triangle(mempool, tri_vertices[i])
        mempool.total_intersection += 1
        if uvt[2] > 0.0 and uvt[2] < nearest_t:
            nearest_t = uvt[2]
            nearest_u = uvt[0]
            nearest_v = uvt[1]
            hit_id = i

    if hit_id >= 0:
        i = mempool.depth
        # store distance
        mempool.hit_t[i] = nearest_t
        # store hit point
        axpy(nearest_t, mempool.ray_d, mempool.ray_o, mempool.hit_p[i])
        # store face normals/tangents/binormals
        copy(mempool.hit_n[i], data.face_normals[hit_id])
        copy(mempool.hit_tn[i], data.face_tangents[hit_id])
        copy(mempool.hit_bn[i], data.face_binormals[hit_id])
        # compute interpolated normal for shading
        tri_interpolation(data.tri_normals[hit_id], nearest_u, nearest_v, mempool.hit_in[i])
        normalize(mempool.hit_in[i])
        # store faceid and material
        mempool.hit_face_id[i] = hit_id
        copy(mempool.hit_material[i], data.face_materials[hit_id])
        mempool.hit_materialtype[i] = data.face_materialtype[hit_id]

        # two-sided intersection
        if dot(mempool.ray_d, mempool.hit_n[i]) > 0:
            mempool.hit_n[i] *= -1.0
        if dot(mempool.ray_d, mempool.hit_in[i]) > 0:
            mempool.hit_in[i] *= -1.0

@numba.njit
def rendering_equation(details, mempool):
    # update ray and compute weakening factor
    update_ray_from_uniform_distribution(mempool)
    weakening_factor = dot(mempool.ray_d, mempool.hit_in[mempool.depth])

    # rendering equation : emittance + (BRDF * incoming * cos_theta / pdf);
    mempool.result *= mempool.hit_material[mempool.depth]
    mempool.result *= INV_PI * weakening_factor * INV_PDF
    recursive_trace(details, mempool)

@numba.njit
def recursive_trace(details, mempool):
    if mempool.depth + 1 >= MAX_DEPTH: # can another hit be allocated ?
        copy(mempool.result, BLACK)
        return

    ray_tri_details(details, mempool)
    if not mempool.valid_hit():
        copy(mempool.result, BLACK)
        return

    if mempool.hit_materialtype[mempool.depth]==LIGHT_MATERIAL_ID:
        mempool.result *= mempool.hit_material[mempool.depth]
        return

    rendering_equation(details, mempool)

@numba.njit
def start_trace(details, mempool):
    ray_tri_details(details, mempool)

    if not mempool.valid_hit():
        copy(mempool.result, BLACK)
        return

    if MAX_DEPTH == 0:
        copy(mempool.result, mempool.hit_material[0])
        mempool.result *= abs(dot(mempool.hit_in[0], mempool.ray_d))
        return

    if mempool.hit_materialtype[0]==LIGHT_MATERIAL_ID:
        copy(mempool.result, mempool.hit_material[0])
        return

    copy(mempool.result, WHITE)

    rendering_equation(details, mempool)

@numba.njit(nogil=True)
def render(image, camera, details, start_time, thread_id=0):
    mempool = jit_core.MemoryPool(NUM_SAMPLES)
    random.seed(RANDOM_SEED)
    row_step = CPU_COUNT
    row_start = thread_id

    for j in range(row_start, camera.height,row_step):
        for i in range(camera.width):
            jj = camera.height-1-j
            ii = camera.width-1-i

            for sx in range(SUPERSAMPLING):
                for sy in range(SUPERSAMPLING):

                    # compute shade
                    c = np.zeros(3)
                    for _ in range(NUM_SAMPLES):
                        mempool.result[0:3] = 0.0
                        camera.get_ray(i, j, sx, sy, mempool)
                        start_trace(details, mempool)
                        mempool.result /= NUM_SAMPLES
                        c += mempool.result
                    clamp(c)
                    c /= (SUPERSAMPLING * SUPERSAMPLING)
                    image[jj, ii] += c

            gamma_correction(image[jj, ii])

        with numba.objmode():
            p = (j+1) / camera.height
            print('. completed : %.2f' % (p * 100.0), ' %')
            if time.time() != start_time:
                t = time.time() - start_time
                estimated_time_left = (1.0 - p) / p * t
                print('    estimated time left: %.2f sec (threadId %d)' % (estimated_time_left, thread_id))

    with numba.objmode():
        print('Total intersections : %d (threadId %d)' % (mempool.total_intersection, thread_id))

    return mempool.total_intersection