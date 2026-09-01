import numpy as np

# 
def make_chain(n):
    # Open-boundary chain with absorbing final state: A -> B -> C -> D
    T = np.eye(n, k=1)
    T[-1, -1] = 1.0
    return T

def make_ring(n):
    # Ring: A -> B -> C -> D -> A
    return np.eye(n, k=1) + np.eye(n, k=-(n-1))

# Brief Test
T_chain = make_chain(4)
T_ring = make_ring(4)
# Looking for rows to sum to 1
print(np.allclose(T_chain.sum(axis=1), 1.0))
print(np.allclose(T_ring.sum(axis=1), 1.0))
print(T_ring)