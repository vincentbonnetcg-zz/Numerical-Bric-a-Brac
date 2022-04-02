"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import IPython.display

import numpy as np
import io
import PIL
import time
from concurrent.futures import ThreadPoolExecutor

import common
from scene import Scene
import jit.pathtracer as pathtracer

@common.timeit
def force_jit(image, camera, details):
    # jit compilation by calling a tiny scene
    width, height = camera.width, camera.height
    camera.set_resolution(2, 2)
    pathtracer.render(image, camera, details, time.time())
    camera.set_resolution(width, height)
    image.fill(0.0)

@common.timeit
def render(image, camera, details):
    futures = []
    with ThreadPoolExecutor(max_workers=pathtracer.CPU_COUNT) as executor:
        start_time = time.time()
        for thread_id in range(pathtracer.CPU_COUNT):
            future = executor.submit(pathtracer.render, image, camera,
                                     details, start_time, thread_id)
            futures.append(future)

    total_intersections = 0
    for future in futures:
        total_intersections += future.result()
    print('Total intersections : %d' % total_intersections)

@common.timeit
def show(image):
    buffer = io.BytesIO()
    PIL.Image.fromarray(np.uint8(image*255)).save(buffer, 'png')
    IPython.display.display(IPython.display.Image(data=buffer.getvalue()))

def main():
    pathtracer.MAX_DEPTH = 5 # max ray bounces
    pathtracer.NUM_SAMPLES = 200 # number of sample per pixel
    pathtracer.RANDOM_SEED = 10
    pathtracer.SUPERSAMPLING = 1
    pathtracer.CPU_COUNT = 6

    scene = Scene()
    #scene.load_original_cornell_box()
    scene.load_teapot_scene()
    details = scene.tri_details()

    camera = scene.camera
    camera.set_resolution(512, 512)
    camera.set_supersampling(pathtracer.SUPERSAMPLING)
    image = np.zeros((camera.height, camera.width, 3))

    # Make sure the pathtracer.CONSTANTS are set before calling jitted functions
    force_jit(image, camera, details)
    render(image, camera, details)
    show(image)

if __name__ == '__main__':
    main()
