import numpy as np
from copy import deepcopy
from random import choice, random
from ot import fused_gromov_wasserstein2

def node_dists(G1, G2, d):
    """
    Return the matrix distance between two graphs

    Args :
    - G1 (nx.Graph) : first graph
    - G2 (nx.Graph) : second graph
    - d : real valued distance function
    """
    M = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))

    for i, u in enumerate(G1.nodes):
        for j, v in enumerate(G2.nodes):
            M[i, j] = d(G1.nodes[u]['x'], G2.nodes[v]['x'])
    return M

def greedy_random_transport_plan(p, q, e=1e-10):
    """
    Computes a random transport plan between two distributions with a greedy method.

    Args :
    - p : source histogram
    - q : target histogram
    - e (float, optionnal : default = 1e-10) : threshold for ending a transport

    Output :
    - T (np.array) : transport matrix
    """
    n = p.shape[0]
    m = q.shape[0]
    e /= n
    tp = deepcopy(p)
    tq = deepcopy(q)
    ixp = [i for i in range(n)]
    ixq = [i for i in range(m)]
    T = np.zeros((n, m))
    
    while len(ixp) > 0 and len(ixq) > 0:
        ip = choice(ixp)
        iq = choice(ixq)

        if tp[ip] < e or tq[iq] < e:
            #We transfer everything that we can
            t = min(tp[ip], tq[iq])
        else:
            t = random()*min(tp[ip], tq[iq])

        tp[ip] -= t
        tq[iq] -= t
        T[ip, iq] += t

        if tp[ip] == 0:
            ixp.pop(ixp.index(ip))
        if tq[iq] == 0:
            ixq.pop(ixq.index(iq))

    return T

def fgw(C1, C2, M, h1, h2, alpha=0.5, Niter=100, verbose=True):
    """
    Compute FGW distance

    Args :
    - C1 (np.array) : first structre matrix
    - C2 (np.array) : second structre matrix
    - M (np.array) : distance matrix
    - h1 (array like) : first histogram
    - h2 (array like) : second histogram
    - alpha (float, default : 0.5) : alpha coefficient for FGW
    - Niter (int, default : 100) : Number of iterations starting from a greedy random transport plan
    - verbose (boolean, default : True) : prints information about the computation (number of iteration and if a greedy transport plan is not correct)

    Output : 
    - fgw (float) : FGW distance between the two structured data
    - Tmin (np.array) : optimal coupling between the two structured data
    """
    def printv(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    #First try with the "hard coded" transport plan
    min_dist, log = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha, log=True)
    Tmin = log['T']
    
    for i in range(Niter):
        printv(f"Iteration {i+1}/{Niter}", end="\r")
        G0 = greedy_random_transport_plan(h1, h2)
        try:
            f, log = fused_gromov_wasserstein2(M, C1, C2, h1, h2, alpha=alpha, G0=G0, log=True)
        except AssertionError:
            printv("A greedy generated transport plan was not exact")
            continue
        if (f < min_dist or min_dist < 0) and f >= 0:
            min_dist = f
            Tmin = log['T']
    return min_dist, Tmin