import sys
import numpy as np
import jax.numpy as jnp
from jax import jit
import jaxopt

num_peaks = float(sys.argv[1])

data = np.loadtxt(sys.stdin)
ppm = data[:, 0]
intensity = data[:, 1]

@jit
def lorentzian(x, A, x0, gamma):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

@jit
def model(params, x):
    p = params.reshape(-1, 3)
    
    A = p[:, 0, None]
    x0 = p[:, 1, None]
    gamma = p[:, 2, None]
    
    x_reshaped = x[None, :]
    
    peaks = A * (gamma**2 / ((x_reshaped - x0)**2 + gamma**2))
    return jnp.sum(peaks, axis=0)

@jit
def loss_function(params, x, y):
    current_fit = model(params, x)
    return jnp.sum((current_fit - y)**2)

p0 = jnp.array([50, -61.75, 0.2, 20, -62.75, 0.15, 10, -61.75, 0.8, 10, -62.75, 0.8])

solver = jaxopt.LBFGSB(fun=loss_function)

lower_bounds = jnp.full_like(p0, -jnp.inf)
upper_bounds = jnp.full_like(p0, jnp.inf)
bounds = (lower_bounds, upper_bounds)

result = solver.run(p0, bounds=bounds, x=ppm, y=intensity)
popt = result.params

lorentzian_line = model(popt, ppm)
residuals = intensity - lorentzian_line

columns = [ppm, lorentzian_line, residuals]
for i in range(0, len(popt), 3):
    individual = lorentzian(ppm, popt[i], popt[i+1], popt[i+2])
    columns.append(individual)

result = np.column_stack(columns)

np.savetxt(sys.stdout, result, fmt='%.6f')
