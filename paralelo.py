import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

NI = 2
NJ = 2
TI = 3
TJ = 3
RESOLUTIONI = 3
RESOLUTIONJ = 3

def spline_knots(knots, n, t):
    for j in range(n + t + 1):
        if j < t:
            knots.append(0)
        elif j <= n:
            knots.append(j - t + 1)
        else:
            knots.append(n - t + 2)

def spline_blend(k, t, u, v):
    if t == 1:
        return 1.0 if u[k] <= v < u[k + 1] else 0.0
    else:
        value = 0.0
        if u[k + t - 1] != u[k]:
            value += ((v - u[k]) / (u[k + t - 1] - u[k])) * spline_blend(k, t - 1, u, v)
        if u[k + t] != u[k + 1]:
            value += ((u[k + t] - v) / (u[k + t] - u[k + 1])) * spline_blend(k + 1, t - 1, u, v)
        return value

inp = np.zeros((NI + 1, NJ + 1, 3))
outp = np.zeros((RESOLUTIONI, RESOLUTIONJ, 3))
knots_i = []
knots_j = []

np.random.seed(42)
for i in range(NI + 1):
    for j in range(NJ + 1):
        inp[i, j] = np.array([i, j, np.random.uniform(-1.0, 1.0)])

increment_i = (NI - TI + 2) / RESOLUTIONI
increment_j = (NJ - TJ + 2) / RESOLUTIONJ

spline_knots(knots_i, NI, TI)
spline_knots(knots_j, NJ, TJ)

def compute_row(interval_i):
    row_result = np.zeros((RESOLUTIONJ, 3))
    interval_j = 0.0
    for j in range(RESOLUTIONJ):
        for ki in range(NI + 1):
            for kj in range(NJ + 1):
                bi = spline_blend(ki, TI, knots_i, interval_i)
                bj = spline_blend(kj, TJ, knots_j, interval_j)
                row_result[j] += inp[ki, kj] * (bi * bj)
        interval_j += increment_j
    return row_result

# Parallel computation
results = Parallel(n_jobs=-1)(delayed(compute_row)(interval_i) for interval_i in np.arange(0, RESOLUTIONI * increment_i, increment_i))

outp = np.array(results)

x = outp[:, :, 0]
y = outp[:, :, 1]
z = outp[:, :, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.scatter(
    inp[:, :, 0].flatten(),
    inp[:, :, 1].flatten(),
    inp[:, :, 2].flatten(),
    color='red',
    marker='o',
    label='Control Points'
)
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()