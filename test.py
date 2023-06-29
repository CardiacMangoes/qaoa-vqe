import multiprocessing
import numpy as np
import scipy
import time

def square(x):
    return np.sum(x ** 2)

def minimize(x0):
    res = scipy.optimize.minimize(
                                fun = lambda x: square(*x), 
                                x0 = [x0],
                                tol = 1e-6)
    return res.x

def run():
    start = time.time()

    cpu_count = multiprocessing.cpu_count()

    initial_vals = (np.random.rand(144) - 0.5) * 1e20

    # p = multiprocessing.Pool(cpu_count)
    # results = p.map(minimize,initial_vals)

    results = []
    for i in initial_vals:
        results.append(minimize(i))

    print(len(results))
    print(time.time() - start)


if __name__ == '__main__':
    
    run()