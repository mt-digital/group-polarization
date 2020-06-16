import matplotlib.pyplot as plt
import numpy as np

from experiments.within_box import BoxedCavesExperiment

# Set up experiment and iterate.
n_caves = 20
n_per_cave = 5
S = 0.5
bce = BoxedCavesExperiment(n_caves, n_per_cave, S, K=2)

bce.iterate(5)

# Extract initial and final opinion coordinates for plotting.
init_coords = np.array(bce.history['coords'][0])
final_coords = np.array(bce.history['final coords'])

xi = init_coords[:, 0]; yi = init_coords[:, 1]
xf = final_coords[:, 0]; yf = final_coords[:, 1]

# Plot the results.
plt.figure()
plt.plot(xi, yi, 'o', label='Initial opinions')
plt.plot(xf, yf, 'o', label='Opinions after five iterations')
lim = [-.55, .55]; plt.xlim(lim); plt.ylim(lim)
plt.axis('equal')
plt.legend()
plt.savefig('simple_experiment.png', dpi=300)
