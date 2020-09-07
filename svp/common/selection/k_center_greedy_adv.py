import numpy as np
from tqdm import tqdm
import scipy.spatial.distance  # type: ignore



def k_center_greedy_minmax_adv_slow(X, s, b, g):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X representing the existing pool
    - b: int, selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0] // g
    p = np.setdiff1d(np.arange(n), s, assume_unique=True).tolist()  # pool indices
    sel = list(s)

    for i in range(b):
        sel_ind = np.repeat(sel, g) * g
        sel_ind += np.tile(np.arange(g), len(sel))
        p_ind = np.repeat(p, g) * g
        p_ind += np.tile(np.arange(g), len(p))
        D = scipy.spatial.distance.cdist(X[sel_ind], X[p_ind], metric='euclidean')  # shape (|s|,|p|)
        j = np.argmax(np.min(D, axis=0)) // g
        u = p[j]
        sel.append(u)
        p.pop(j)

    return np.asarray(sel[-b:])

def k_center_greedy_avg_adv_slow(X, s, b, g):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X representing the existing pool
    - b: int, selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0] // g
    p = np.setdiff1d(np.arange(n), s, assume_unique=True).tolist()  # pool indices
    sel = list(s)

    for i in range(b):
        sel_ind = np.repeat(sel, g) * g
        sel_ind += np.tile(np.arange(g), len(sel))
        p_ind = np.repeat(p, g) * g
        p_ind += np.tile(np.arange(g), len(p))
        D = scipy.spatial.distance.cdist(X[sel_ind], X[p_ind], metric='euclidean')  # shape (|s|,|p|)
        avg_vec1 = np.eye(len(sel))
        avg_vec1 = np.repeat(avg_vec1, g, axis=1)
        avg_vec2 = np.eye(len(p))
        avg_vec2 = np.repeat(avg_vec2, g, axis=1)
        D = np.dot(avg_vec1, np.dot(D, avg_vec2.T)) / (g*g)
        j = np.argmax(np.min(D, axis=0))
        u = p[j]
        sel.append(u)
        p.pop(j)

    return np.asarray(sel[-b:])

def k_center_greedy_avg_adv_slower(X, s, b, g):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X representing the existing pool
    - b: int, selection budget

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0] // g
    p = np.setdiff1d(np.arange(n), s, assume_unique=True).tolist()  # pool indices
    sel = list(s)

    for i in range(b):
        sel_ind = np.repeat(sel, g) * g
        sel_ind += np.tile(np.arange(g), len(sel))
        p_ind = np.repeat(p, g) * g
        p_ind += np.tile(np.arange(g), len(p))
        D = scipy.spatial.distance.cdist(X[sel_ind], X[p_ind], metric='euclidean')  # shape (|s|,|p|)
        new_D = np.zeros([len(sel), len(p)], dtype=np.float32)
        for mm in range(len(sel)):
            for nn in range(len(p)):
                new_D[mm, nn] = np.mean(D[mm*g:(mm+1)*g, nn*g:(nn+1)*g])
        j = np.argmax(np.min(new_D, axis=0))
        u = p[j]
        sel.append(u)
        p.pop(j)

    return np.asarray(sel[-b:])

def k_center_greedy_minmax_adv(X, s, b, g):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X that have already been selected
    - b: int, new selection budget
    - g: int, number of iterations of PGD

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0] // g
    p = np.setdiff1d(np.arange(n), s, assume_unique=True)  # pool indices
    sel = np.empty(b, dtype=np.int64)

    sl = len(s)
    D = np.zeros([(sl + b) * g, len(p) * g], dtype=np.float32)
    ind_s = np.repeat(s, g) * g
    ind_s += np.tile(np.arange(g), sl)
    ind_p = np.repeat(p, g) * g
    ind_p += np.tile(np.arange(g), len(p))

    D[:sl*g] = scipy.spatial.distance.cdist(X[ind_s], X[ind_p], metric='euclidean')  # shape (|s|,|p|)
    mins = np.min(D[:sl*g], axis=0)  # vector of length |p|
    cols = np.ones(len(p), dtype=bool)  # columns still in use

    for i in tqdm(range(b), desc="Greedy k-Centers"):
        j_ = np.argmax(mins)
        j = j_ // g
        u = p[j]
        sel[i] = u

        if i == b - 1:
            break

        mins[j*g:(j+1)*g] = -1
        cols[j] = False

        # compute dist between selected point and remaining pool points
        r = sl + i
        p_ = p[cols]
        ind_p = np.repeat(p_, g) * g
        ind_p += np.tile(np.arange(g), len(p_))
        D[r * g: (r+1)*g, np.repeat(cols, g)] = scipy.spatial.distance.cdist(X[u*g:(u+1)*g], X[ind_p])
        mins = np.minimum(mins, np.min(D[r*g:(r+1)*g], axis=0))

    return sel

def k_center_greedy_avg_adv(X, s, b, g):
    '''
    Args
    - X: np.array, shape [n, d]
    - s: list of int, indices of X that have already been selected
    - b: int, new selection budget
    - g: int, number of iterations of PGD

    Returns: np.array, shape [b], type int64, newly selected indices
    '''
    n = X.shape[0] // g
    p = np.setdiff1d(np.arange(n), s, assume_unique=True)  # pool indices
    sel = np.empty(b, dtype=np.int64)

    sl = len(s)
    D_ = np.zeros([(sl + b) * g, len(p) * g], dtype=np.float32)
    D = np.zeros([(sl + b), len(p)], dtype=np.float32)
    ind_s = np.repeat(s, g) * g
    ind_s += np.tile(np.arange(g), sl)
    ind_p = np.repeat(p, g) * g
    ind_p += np.tile(np.arange(g), len(p))

    D_[:sl*g] = scipy.spatial.distance.cdist(X[ind_s], X[ind_p], metric='euclidean')  # shape (|s|,|p|)

    avg_vec1 = np.eye(sl)
    avg_vec1 = np.repeat(avg_vec1, g, axis=1)
    avg_vec2 = np.eye(len(p))
    avg_vec2 = np.repeat(avg_vec2, g, axis=1)

    D[:sl] = np.dot(avg_vec1, np.dot(D_[:sl*g], avg_vec2.T)) / (g*g)

    mins = np.min(D[:sl], axis=0)  # vector of length |p|
    cols = np.ones(len(p), dtype=bool)  # columns still in use

    for i in tqdm(range(b), desc="Greedy k-Centers"):
        j = np.argmax(mins)
        u = p[j]
        sel[i] = u

        if i == b - 1:
            break

        mins[j] = -1
        cols[j] = False

        # compute dist between selected point and remaining pool points
        r = sl + i
        p_ = p[cols]
        ind_p = np.repeat(p_, g) * g
        ind_p += np.tile(np.arange(g), len(p_))
        D_[r * g: (r+1)*g, np.repeat(cols, g)] = scipy.spatial.distance.cdist(X[u*g:(u+1)*g], X[ind_p])
        avg_vec1 = np.eye(1)
        avg_vec1 = np.repeat(avg_vec1, g, axis=1)
        avg_vec2 = np.eye(len(p_))
        avg_vec2 = np.repeat(avg_vec2, g, axis=1)
        D[r, cols] = np.dot(avg_vec1, np.dot(D_[r * g: (r+1)*g, np.repeat(cols, g)], avg_vec2.T)) / (g*g)

        mins = np.minimum(mins, D[r])

    return sel


if __name__ == '__main__':
    import time
    for i in range(10):
        g = 4
        n, d = np.random.randint(10, 1000, size=2)
        X = np.random.randn(n, d)
        X = np.repeat(X, g, axis=0)
        noise = np.random.randn(X.shape[0], X.shape[1]) * 0.01
        X += noise
        s0_size = np.random.randint(1, int(n/2))
        s = np.random.choice(n, size=s0_size)
        b = np.random.randint(1, int((n - s0_size) / 2))
        start = time.time()
        fast = k_center_greedy_avg_adv(X, s, b, g)
        fast_time = time.time() - start
        start = time.time()
        slow = k_center_greedy_avg_adv_slow(X, s, b, g)
        slow_time = time.time() - start
        assert np.all(fast == slow)
        print(f'{i}: (n={n}, d={d}, b={b}), fast {fast_time:.2f}, slow {slow_time:.2f}')
