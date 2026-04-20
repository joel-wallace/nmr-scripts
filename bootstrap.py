import sys
import numpy as np
import jax.numpy as jnp
from jax import jit
import jaxopt
import jax
jax.config.update("jax_enable_x64", True)
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

# user guesses
p0 = jnp.array([50, -61.75, 0.2, 20, -62.75, 0.15, 10, -61.75, 0.8, 10, -62.75, 0.8])

# LBFGSB is a gradient descent with box constraints
# it allows setting bounds e.g. so that amplitude is not negative
solver = jaxopt.LBFGSB(fun=loss_function)

lower_bounds = jnp.zeros_like(p0)

# A bounds
lower_bounds = lower_bounds.at[0::3].set(-jnp.inf)

# x0 bounds
lower_bounds = lower_bounds.at[1::3].set(-jnp.inf)

# gamma bounds
lower_bounds = lower_bounds.at[2::3].set(-jnp.inf)

# Upper bounds remain infinite
upper_bounds = jnp.full_like(p0, jnp.inf)

# Final bounds tuple for the solver
bounds = (lower_bounds, upper_bounds)
# initial fit
result = solver.run(p0, bounds=bounds, x=ppm, y=intensity)
# popt stands for optimized parameters
popt = result.params
# draws optimised summed line
lorentzian_line = sum_lorentzians(popt, ppm)
residuals = intensity - lorentzian_line

@jit(static_argnames=['itercount'])
def bootstrap(x, y, itercount, ipopt, residuals, bounds):
    solver = jaxopt.LBFGSB(fun=loss_function)
    x_points = len(x)
    seed = 999
    key = jax.random.key(seed)
    sample_indices = jax.random.randint(key, (itercount, x_points), 0, x_points)
    sample_residuals = residuals[sample_indices]
    sample_y = y + sample_residuals
    # results = jax.vmap(lambda ipopt, bounds, x, y:
    #          solver.run(ipopt, bounds=bounds, x=x, y=y),
    #          in_axes=(None, None, None, 0))(ipopt, bounds, x, sample_y)
    # return results.params
    
    def fit_single(y_sample):
        return solver.run(ipopt, bounds=bounds, x=x, y=y_sample).params

    results = jax.lax.map(fit_single, sample_y)
    return results

test = bootstrap(ppm, lorentzian_line, 1000, popt, residuals, bounds)

# 
# calculate std error of the values
# boom done
# columns for output file
columns = [ppm, lorentzian_line, residuals]
# draws the individual lines
for i in range(0, len(popt), 3):
    individual = lorentzian(ppm, popt[i], popt[i+1], popt[i+2])
    columns.append(individual)

# output
result = np.column_stack(columns)
np.savetxt(sys.stdout, result, fmt='%.6f')
print(test)
