import numpy as np
import matplotlib.pyplot as plt

# 
def make_chain(n):
    # Open-boundary chain with absorbing final state: A -> B -> C -> D
    T = np.eye(n, k=1)
    T[-1, -1] = 1.0
    return T

def make_ring(n):
    # Ring: A -> B -> C -> D -> A
    return np.eye(n, k=1) + np.eye(n, k=-(n-1))

def successor_representation(T, gamma=0.9):
    n = T.shape[0]
    I = np.eye(n)
    return np.linalg.inv(I - gamma * T)



# Brief Test
T_chain = make_chain(4)
T_ring = make_ring(4)

M_ring = successor_representation(T_ring, gamma=0.9)
print(M_ring)
print(np.linalg.eigvals(M_ring))

vals_T, vecs_T = np.linalg.eig(T_ring)
vals_M, vecs_M = np.linalg.eig(M_ring)

# sort both by real part of eigenvalue, descending, so columns line up
order_T = np.argsort(vals_T.real)[::-1]
order_M = np.argsort(vals_M.real)[::-1]

n = T_ring.shape[0]
x = np.arange(1, n + 1)

fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), sharex=True)

for col in range(n):
    idx = order_T[col]
    axes[0, col].plot(x, vecs_T[:, idx].real, 'o-', color='tab:blue')
    axes[0, col].axhline(0, color='gray', lw=0.5)
    axes[0, col].set_title(f"λ={vals_T[idx]:.2f}")

    idx = order_M[col]
    axes[1, col].plot(x, vecs_M[:, idx].real, 'o-', color='tab:orange')
    axes[1, col].axhline(0, color='gray', lw=0.5)
    axes[1, col].set_title(f"λ={vals_M[idx]:.2f}")

axes[0, 0].set_ylabel("T (ring)")
axes[1, 0].set_ylabel("SR (ring)")
fig.suptitle("Ring: eigenvectors of T vs. SR")
fig.tight_layout()
plt.show()