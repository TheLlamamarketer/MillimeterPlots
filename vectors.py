import numpy as np
import matplotlib.pyplot as plt


for n in range(10):
    for k in range(n+1):
        print((k, n-k))






v = np.array([0, 10])
v = v / np.linalg.norm(v)
u = np.array([1.5, 0.6])

def conjugate(v):
    return np.array([v[0], -v[1]])

u_con = conjugate(u)

def draw_vector(v, label=None, color=None):
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, label=label, color=color)
    plt.scatter(v[0], v[1], alpha=0)

def vec_matrix(v):
    return np.array([[v[0], v[1]], [-v[1], v[0]]])

def vector_results(u, v):
    u_v = vec_matrix(u) @ vec_matrix(v)
    return u_v[0]


plt.figure(figsize=(6, 6), )
plt.plot([0, 3*v[0]], [0, 3*v[1]], 'r-', label='v')

draw_vector(u, label='u', color='g')
draw_vector(u_con, label='$\\bar{u}$', color='b')
draw_vector(vector_results(vector_results(v,v), u_con), label='u * v', color='m')

plt.axis('equal')
plt.legend()
plt.grid()
plt.show()