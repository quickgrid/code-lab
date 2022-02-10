import time

from numba import njit, prange
import numpy as np
import fire


class A:
    def __init__(self):
        super(A, self).__init__()

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def do_something_numba():
        k = np.ones(shape=(10000000,))
        for i in prange(10000000):
            k[i] += np.random.randint(1000)
        return k

    @staticmethod
    def do_something():
        k = np.ones(shape=(10000000,))
        for i in range(10000000):
            k[i] += np.random.randint(1000)
        return k

    @staticmethod
    def time_implementations():
        start_time = time.time()
        k = A.do_something_numba()
        print(f'NUMBA TIME: {time.time() - start_time}')
        print(k.shape)

        start_time = time.time()
        k = A.do_something()
        print(f'PYTHON TIME: {time.time() - start_time}')
        print(k.shape)


if __name__ == '__main__':
    # fire.Fire(A)
    A.time_implementations()

