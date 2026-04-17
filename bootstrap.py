import sys
import numpy as np
import jax.numpy as jnp
from jax import jit
import jaxopt

# command line argument
num_peaks = float(sys.argv[1])

# load from standard input (pipe operator)
data = np.loadtxt(sys.stdin)
ppm = data[:, 0]
intensity = data[:, 1]

# lorentzian equation
@jit
def lorentzian(x, A, x0, gamma):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

# sums lorentzians in a vectorized manner
@jit
def sum_lorentzians(params, x):
    p = params.reshape(-1, 3)
    
    A = p[:, 0, None]
    x0 = p[:, 1, None]
    gamma = p[:, 2, None]
    
    x_reshaped = x[None, :] 
    # these are vectors (arrays), not single lorentzians
    peaks = lorentzian(x_reshaped, A, x0, gamma)
    return jnp.sum(peaks, axis=0)

# aim to minimise residuals (return value)
@jit
def loss_function(params, x, y):
    current_fit = sum_lorentzians(params, x)
    return jnp.sum((current_fit - y)**2)

# initial guesses
p0 = jnp.array([50, -61.75, 0.2, 20, -62.75, 0.15, 10, -61.75, 0.8, 10, -62.75, 0.8])

# LBFGSB is a gradient descent with box constraints
# it allows setting bounds e.g. so that amplitude is not negative
solver = jaxopt.LBFGSB(fun=loss_function)

# placeholder infinite bounds until I implement it properly
lower_bounds = jnp.full_like(p0, -jnp.inf)
upper_bounds = jnp.full_like(p0, jnp.inf)
bounds = (lower_bounds, upper_bounds)

result = solver.run(p0, bounds=bounds, x=ppm, y=intensity)

# popt stands for optimized parameters
popt = result.params

# draws optimised summed line
lorentzian_line = sum_lorentzians(popt, ppm)
residuals = intensity - lorentzian_line
# columns for output file
columns = [ppm, lorentzian_line, residuals]
# draws the individual lines
for i in range(0, len(popt), 3):
    individual = lorentzian(ppm, popt[i], popt[i+1], popt[i+2])
    columns.append(individual)

# output
result = np.column_stack(columns)
np.savetxt(sys.stdout, result, fmt='%.6f')
