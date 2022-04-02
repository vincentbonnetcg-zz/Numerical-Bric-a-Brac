"""
@author: Vincent Bonnet
@description : functions to create geometry
"""

import numpy as np
from jit import maths as jit_maths

def create_polygon_soup(num_triangles):
    tv = np.zeros((num_triangles, 3, 3), dtype=float) # triangle vertices
    tn = np.zeros((num_triangles, 3, 3), dtype=float) # triangle normal
    return tv, tn

def create_tri_quad(quad_corners):
    tv, tn = create_polygon_soup(2)
    np.copyto(tv[0][0], quad_corners[0])
    np.copyto(tv[0][1], quad_corners[1])
    np.copyto(tv[0][2], quad_corners[2])
    np.copyto(tv[1][0], quad_corners[0])
    np.copyto(tv[1][1], quad_corners[2])
    np.copyto(tv[1][2], quad_corners[3])
    n0 = jit_maths.cross(tv[0][2]-tv[0][0], tv[0][1]-tv[0][0])
    n1 = jit_maths.cross(tv[1][2]-tv[1][0], tv[1][1]-tv[1][0])
    jit_maths.normalize(n0)
    jit_maths.normalize(n1)
    np.copyto(tn[0][0], n0)
    np.copyto(tn[0][1], n0)
    np.copyto(tn[0][2], n0)
    np.copyto(tn[1][0], n1)
    np.copyto(tn[1][1], n1)
    np.copyto(tn[1][2], n1)
    return tv, tn

def load_obj(path, scale = 1.0, translate = [0.,0.,0.], smooth_normal = False):
    # this is not a fully functional obj-reader !
    with open(path) as f:
        content = f.readlines()

    # gather vertices/normals/faces
    vertices = []
    normals = []
    tri_vertex_ids = []
    tri_normal_ids = []
    for line in content:
        tokens = line.split()
        if len(tokens) == 0:
            continue

        if tokens[0]=='v': # x y z w
            values = np.asarray([float(v) for v in tokens[1:4]])
            vertices.append(values)
        elif tokens[0]=='vn': # x y z
            values = [float(v) for v in tokens[1:4]]
            normals.append(values)
        elif tokens[0]=='f': # f  v1/vt1/vn1   v2/vt2/vn2   v3/vt3/vn3 . . .
            nVertexPerFace = len(tokens) - 1
            validFace = (nVertexPerFace==3 or nVertexPerFace==4)
            assert validFace, "face should be tri/quad"

            '''
            if nVertexPerFace==4:
                pass
                # triangulation code
                # https://notes.underscorediscovery.com/obj-parser-easy-parse-time-triangulation/
            '''
            nTriPerFace = nVertexPerFace - 2
            for t in range(nTriPerFace):
                tv = []
                tn = []
                vtx_ids = [0,1+t,2+t]
                for vtx in vtx_ids:
                    v_vt_vn = tokens[vtx+1].split('/')
                    if (int(v_vt_vn[0]) < 0):
                        tv.append(int(v_vt_vn[0]))
                    else:
                        tv.append(int(v_vt_vn[0])-1) # index start at 1 !

                    if int(v_vt_vn[2]) < 0:
                        tn.append(int(v_vt_vn[2]))
                    else:
                        tn.append(int(v_vt_vn[2])-1) # index start at 1 !

                tri_vertex_ids.append(tv)
                tri_normal_ids.append(tn)

    num_triangles = len(tri_vertex_ids)
    num_vertices = len(vertices)
    # rescale and center the vertices
    vertices = np.asarray(vertices)
    normals = np.asarray(normals)
    # rescale
    vertices *= scale
    # center
    min_v = np.min(vertices, axis=0)
    max_v = np.max(vertices, axis=0)
    center = (max_v + min_v)*0.5
    vertices -= center
    # rotation around y and x axis
    #ry = 3.1415 / 1.1
    #rx = -3.1415 / 10
    #mRotY = np.asarray([[np.cos(ry), 0, np.sin(ry)],[0,1,0],[-np.sin(ry), 0, np.cos(ry)]])
    #mRotX = np.asarray([[1, 0, 0],[0,np.cos(rx),-np.sin(rx)],[0, np.sin(rx), np.cos(rx)]])
    #vertices = vertices.dot(mRotY)
    #vertices = vertices.dot(mRotX)
    #normals = normals.dot(mRotY)
    #normals = normals.dot(mRotX)
    # translate
    vertices += np.array(translate)

    if not smooth_normal:
        tv, tn = create_polygon_soup(num_triangles)
        for i in range(num_triangles):
            for vtx in range(3):
                tv[i][vtx] = vertices[tri_vertex_ids[i][vtx]]
                tn[i][vtx] = normals[tri_normal_ids[i][vtx]]
                jit_maths.normalize(tn[i][vtx])

        return tv, tn

    triangle_normals = np.zeros((num_triangles, 3))
    for i in range(num_triangles):
        #for vtx in range(3):
        #    triangle_normals[i] += normals[tri_normal_ids[i][vtx]]
        e1 = vertices[tri_vertex_ids[i][1]] - vertices[tri_vertex_ids[i][0]]
        e2 = vertices[tri_vertex_ids[i][2]] - vertices[tri_vertex_ids[i][0]]
        triangle_normals[i] = np.cross(e1, e2)
        jit_maths.normalize(triangle_normals[i])


    vertex_normals = np.zeros((num_vertices, 3))
    for i in range(num_triangles):
        for vtx in range(3):
            vertex_normals[tri_vertex_ids[i][vtx]] += triangle_normals[i]
    for i in range(num_vertices):
        jit_maths.normalize(vertex_normals[i])

    # create the triangles
    tv, tn = create_polygon_soup(num_triangles)
    for i in range(num_triangles):
        for vtx in range(3):
            tv[i][vtx] = vertices[tri_vertex_ids[i][vtx]]
            tn[i][vtx] = vertex_normals[tri_vertex_ids[i][vtx]]

    return tv, tn
