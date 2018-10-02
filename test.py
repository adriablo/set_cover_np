
import numpy as np
from timer import Timer


def sparsify(M, n):
    # sparsify
    for i in range(n):
        print('sparsify ({})'.format(i))
        S = np.random.randint(0, high=2, size=M.shape, dtype=np.uint8)
        M = np.multiply(M, S)
    return M


def make_matrix(C, G, spars=0):
    """
    TODO: IMPLEMENT: import the data matrix
    M[C,G] = 0 or 1
    """
    # C = n_candid = 15
    # G = n_item = 9
    # M = np.zeros((C, G), dtype=np.uint8)
    # OR, faster and not so safe:
    # M = np.empty((C, G), dtype=np.uint8)
    # TODO: read the values of M ...

    # DEBUG: random init (zero or one)
    M = np.random.randint(0, high=2, size=(C, G), dtype=np.uint8)
    M = sparsify(M, spars)
    # save M for inspection, debug
    # N = np.copy(M)
    # ~ DEBUG
    return M



def test(sc, M, n_thread=8, col_sum_dtype=np.float32):
    with Timer() as t:
        x = sc.run(M, n_thread, col_sum_dtype)
    print('time: {}'.format(t.interval))
    return x
