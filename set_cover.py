
import numpy as np
from multiprocessing.pool import ThreadPool


"""
TODO:
* Using float32 type instead of uint8 speeds up by at least 2x, but uses 4x memory!!
* All other int and float types are slower

OpenBLAS Setting the number of threads: DOES NOT WORK!
https://github.com/xianyi/OpenBLAS#set-the-number-of-threads-with-environment-variables
--
export OPENBLAS_NUM_THREADS=4
export GOTO_NUM_THREADS=4
export OMP_NUM_THREADS=4
--

Experiments
--
import set_cover as sc
reload(sc)

C = 10**5
G = 10**3
M = sc.make_matrix(C, G, 2)

x = sc.run(M)
print sc.check_solution(M, x[0])

solution size ~= 100
Time ~= 2 minutes
"""


def col_sum_thread(M, col_sum, row_start, row_end):
    """
    Writing directly into destination vector (zero copy)
    """
    print('START thread: {}'.format((row_start, row_end)))
    try:
        col_sum[row_start:row_end] = np.sum(M[row_start:row_end], axis=1)
    except Exception as e:
        print('thread: {}, crashed: '.format((row_start, row_end), e))
    print('END thread: {}'.format((row_start, row_end)))
    return


def parallel_col_sum(M, col_sum, pool):
    n_thread = pool._processes
    # split into n_thread blocks
    n_row, _ = M.shape
    n_row_per_block = int(float(n_row/n_thread)) + 1
    # run each block on a thread
    row_start, row_end = 0, 0
    results = []
    for tx in range(n_thread):
        row_start = row_end
        row_end = row_start + n_row_per_block
        if row_end > n_row:
            row_end = n_row
        # print(row_start, row_end)
        # pool.apply_async(col_sum_thread, (M, row_start, row_end), callback=collect_result)
        res = pool.apply_async(col_sum_thread, (M, col_sum, row_start, row_end))
        results.append(res)
    for res in results:
        res.wait()
    return col_sum


def run(M, n_thread=8, col_sum_dtype=np.float32):
    """
    TODO:
    * Using float32 type instead of uint8 speeds up by at least 2x, but uses 4x memory!!
    * All other int and float types are slower
    --    
    # DO NOT COPY here: takes up a lot of memory
    # Instead, create M directly as np.float32
    M = M.astype(np.float32)
    --
    col_sum_dtype can be np.uint if M is of type(s) np.uint
    --
    """
    C, G = M.shape
    # is it coverable? is there at least a 1 in every column?
    # sum over the C axis (rows) - collapse the first axis (axis 0)
    vg = np.sum(M, axis=0)
    if min(vg) <= 0:
        print('not coverable!')
        return M, vg

    # START

    # solution = set()
    # make the solution a list to show progression order
    solution = []
    cover = np.zeros(G)

    pool = ThreadPool(processes=n_thread)
    col_sum = np.zeros(C, dtype=col_sum_dtype)

    print('START')
    while True:
        # sum over the G axis (columns) - collapse the second axis (axis 1)

        # single-thread
        # col_sum = np.sum(M, axis=1)

        # ,OR: multi-threaded:
        # it's faster to re-use col_sum (don't allocate every time)
        parallel_col_sum(M, col_sum, pool)

        # position of largest item (first occurrence if many)
        # TODO: can pick a different occurrence, to find a different solution
        pmax = np.argmax(col_sum)

        # add candidate to the solution
        print('add to solution: {}'.format(pmax))
        assert pmax not in solution
        solution.append(pmax)
        print('solution size: {}'.format(len(solution)))

        # remember candidate vector (of items)
        vg = M[pmax]
        cover += vg

        print('# of items left to cover: {}'.format(G - np.sum(cover)))

        # update M - remove columns of items found in the candidate (row)
        # find indexes of columns where candidate row has ones (1)
        # first generate all column indexes and add +1, (so we can select even position 0 with a bitmask vector),
        # then multiply by the bitmask vector, and adjust back by adding -1
        xv = (np.arange(G, dtype=np.int) + 1) * vg.astype(np.uint8)
        # select positions that were not masked out
        x = xv[np.nonzero(xv)] - 1
        # print('x: {}'.format(x))
        # zero out matched item columns
        M[:, x] = 0
        # print(M)

        # Note: this operation overwrites the contents of M,
        # which will be all-zero at the end

        # TODO: a further option is to shrink the matrix by removing the zeroed-out columns, by calling np.delete()
        # However, this will scramble (shift) the row indexes, so we need to keep track of original indexes after resize

        # until finished
        if np.sum(cover) >= G:
            break
    # ~ while

    #
    pool.close()
    pool.join()

    print solution
    return solution, M, cover


def check_solution(M, s):
    """
    TODO: verify that all the columns (items) in M
    are covered at least once using all the rows (candidates) in the Solution s
    """
    # is there at least a 1 in every column?
    # sum the solution rows over the C axis (rows) - collapse the first axis (axis 0)
    vg = np.sum(M[np.array(s),], axis=0)
    print(vg)
    # check that each value is non-zero (>= 1 coverage),
    # ie # of non-zeros == # of columns (items)
    return np.count_nonzero(vg) == M.shape[1]
