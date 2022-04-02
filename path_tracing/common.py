"""
@author: Vincent Bonnet
@description : useful functions
"""

import functools
import time

def timeit(method):
    @functools.wraps(method)
    def execute(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('timeit : %s %.3f sec' % (method.__name__, elapsed_time))
        return result

    return execute

