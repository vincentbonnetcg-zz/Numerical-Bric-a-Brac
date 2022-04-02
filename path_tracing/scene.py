"""
@author: Vincent Bonnet
@description : objects to describe a scene
"""

import math
import numpy as np
import geometry
from jit import core as jit_core
from jit import maths as jit_math

class Material:
    def __init__(self, material = [1,1,1], mtype=0):
        self.material = np.asarray(material)
        self.materialtype = mtype # 0 reflectance, 1 : emittance

class TriangleSoup():
    def __init__(self, tri_vertices, tri_normals, material):
        self.tv = tri_vertices
        self.n = tri_normals  # triangle normals
        self.t = None # triangle tangents
        self.b = None # triangle binormals
        self.material = material

    def num_triangles(self):
        return len(self.n)

class Scene:
    def __init__(self):
        self.objects = []
        self.camera = jit_core.Camera(320, 240)

    def add_cornell_box(self, large_light=False):
        white = [1,1,1]
        red = [0.57,0.025,0.025]
        green = [0.025,0.236,0.025]
        light_intensity = 10.0
        light_colour = [1*light_intensity,0.73*light_intensity,0.4*light_intensity]
        # From http://www.graphics.cornell.edu/online/box/data.html
        quad_v = [] # quad vertices
        quad_m = [] # quad material
        # floor
        quad_v.append([[552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2]])
        quad_m.append([white, 0])
        # left wall
        quad_v.append([[552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0]])
        quad_m.append([red, 0])
        # right wall
        quad_v.append([[0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2]])
        quad_m.append([green, 0])
        # back wall
        quad_v.append([[549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2]])
        quad_m.append([white, 0])
        # ceiling (large light)
        quad_v.append([[556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0]])
        quad_m.append([white, int(large_light)])
        # small light
        if large_light==False:
            # added an offset from the cornell box from 548.8 to 548
            quad_v.append([[343,548.79,227],[343,548.79,332],[213,548.79,332],[213,548.79,227]])
            quad_m.append([light_colour, 1])
        # add floor/ceiling/walls
        for i in range(len(quad_v)):
            tv, tn = geometry.create_tri_quad(quad_v[i])
            material = Material(quad_m[i][0], quad_m[i][1])
            self.objects.append(TriangleSoup(tv, tn, material))
        # set camera
        np.copyto(self.camera.origin, [278, 273, -800])
        self.camera.dir_z = 1.0
        focal_length = 35 # in mm
        sensor_size = 25 # in mm (sensor width and height)
        self.camera.fovx = math.atan(sensor_size*0.5/focal_length) * 2

    def load_original_cornell_box(self):
        white = [1,1,1]
        # From http://www.graphics.cornell.edu/online/box/data.html
        self.add_cornell_box()
        # short block
        quad_v = [] # quad vertices
        quad_m = [] # quad material
        quad_v.append([[130,165,65],[82,165,225],[240,165,272],[290,165,114]])
        quad_m.append([white, 0])
        quad_v.append([[290,0,114],[290,165,114],[240,165,272],[240,0,272]])
        quad_m.append([white, 0])
        quad_v.append([[130,0,65],[130,165,65],[290,165,114],[290,0,114]])
        quad_m.append([white, 0])
        quad_v.append([[82,0,225],[82,165,225],[130,165,65],[130,0,65]])
        quad_m.append([white, 0])
        quad_v.append([[240,0,272],[240,165,272],[82,165,225],[82,0,225]])
        quad_m.append([white, 0])
        # tall block
        quad_v.append([[423,330,247],[265,330,296],[314,330,456],[472,330,406]])
        quad_m.append([white, 0])
        quad_v.append([[423,0,247],[423,330,247],[472,330,406],[472,0,406]])
        quad_m.append([white, 0])
        quad_v.append([[472,0,406],[472,330,406],[314,330,456],[314,0,456]])
        quad_m.append([white, 0])
        quad_v.append([[314,0,456],[314,330,456],[265,330,296],[265,0,296]])
        quad_m.append([white, 0])
        quad_v.append([[265,0,296],[265,330,296],[423,330,247],[423,0,247]])
        quad_m.append([white, 0])
        # add blocks
        for i in range(len(quad_v)):
            tv, n = geometry.create_tri_quad(quad_v[i])
            material = Material(quad_m[i][0], quad_m[i][1])
            self.objects.append(TriangleSoup(tv, n, material))

    def load_teapot_scene(self):
        self.add_cornell_box(True)
        # teapot
        blue_grey = [0.44,0.57,0.745]
        scale = 1. / 250.
        translate = [270., 200., 275.]
        #tv, tn = geometry.load_obj('models/teapot.obj', scale, translate, smooth_normal=False)
        tv, tn = geometry.load_obj('models/smooth_teapot.obj', scale, translate, smooth_normal=False)
        #tv, tn = geometry.load_obj('models/sphere.obj', scale, translate, smooth_normal=False)
        self.objects.append(TriangleSoup(tv, tn, Material(blue_grey, 0)))

    def tri_details(self):
        # gather triangles and materials
        num_triangles = 0
        for obj in self.objects:
            num_triangles += obj.num_triangles()

        # numpy dtype to store structure of array
        dtype_dict = {}
        dtype_dict['names'] = ['tri_vertices', 'tri_normals', 'face_tangents',
                               'face_binormals', 'face_materials', 'face_normals', 'face_materialtype']
        dtype_dict['formats'] = []
        dtype_dict['formats'].append((np.float64, (num_triangles,3,3))) # tri_vertices
        dtype_dict['formats'].append((np.float64, (num_triangles,3,3))) # tri_normals
        dtype_dict['formats'].append((np.float64, (num_triangles,3))) # face_tangents
        dtype_dict['formats'].append((np.float64, (num_triangles,3))) # face_binormals
        dtype_dict['formats'].append((np.float64, (num_triangles,3))) # face_materials
        dtype_dict['formats'].append((np.float64, (num_triangles,3))) # face_normals
        dtype_dict['formats'].append((np.int32, num_triangles)) # face_materialtype
        tri_data = np.zeros(1, dtype=np.dtype(dtype_dict, align=True))

        # consolidate triangles in contiguous numpy array
        index = 0
        for tri in self.objects:
            data = tri_data[0]
            for i in range(len(tri.tv)):
                data['tri_vertices'][index] = tri.tv[i]
                data['tri_normals'][index] = tri.n[i]
                data['face_materialtype'][index] = tri.material.materialtype
                data['face_materials'][index] = tri.material.material
                index += 1

        jit_math.compute_face_normals(tri_data[0]['tri_vertices'],
                                      tri_data[0]['face_normals'])

        jit_math.compute_tangents_binormals(tri_data[0]['face_normals'],
                                            tri_data[0]['face_tangents'],
                                            tri_data[0]['face_binormals'])


        print('num_triangles ' , num_triangles)

        return tri_data
