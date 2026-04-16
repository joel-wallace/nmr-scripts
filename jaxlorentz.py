
import sys
import numpy as np
from scipy.optimize import curve_fit
import jax.numpy as jnp
from jax import jit

def lorentzian(x, A, x0, gamma):
    return A * (gamma**2 / ((x - x0)**2 + gamma**2))

def sum_lorentzians(x, *params):
    sum = np.zeros_like(x)
    for i in range(0, len(params), 3):
        A = params[i]
        x0 = params[i+1]
        gamma = params[i+2]
        sum = sum + lorentzian(x, A, x0, gamma)
    return sum   

def loss_function(x, y, *params):
    current_fit = sum_lorentzians(x, params)
    return jnp.sum((current_fit - y)**2)


num_peaks = float(sys.argv[1])

data = np.loadtxt(sys.stdin)
ppm = data[:, 0]
intensity = data[:, 1]


max_data_height = np.max(intensity)
# Height (A) min: 0, max: max_data_height * 2
# Center (x0) min: -70, max: -50
# Width (gamma) min: 0.001, max: 2.0
lower_limit_single = [0, -70, 0.001]
upper_limit_single = [5000, -50, 5.0]

# Multiply these lists by the number of peaks to match p0 length
lower_bounds = lower_limit_single * int(num_peaks)
upper_bounds = upper_limit_single * int(num_peaks)

# initial parameters: height, ppm, width, etc
#p0 = [4000, -61.75, 0.1, 500, -61.75, 0.8, 1500, -62.59, 0.2, 500, -62.6, 0.8]
p0 = [100, -61.9, 0.05, 90, -62.75, 0.05, 20, -61.9, 0.3]#, 20, -62.75, 0.3]
# the function, xdata, ydata, initial estimates, bounds to avoid going wide 
popt, _ = curve_fit(sum_lorentzians, ppm, intensity, p0=p0, bounds=(lower_bounds, upper_bounds))
# jax wants a loss function and returns the value i want to minimize

lorentzian_line = sum_lorentzians(ppm, *popt)

residuals = intensity - lorentzian_line

columns = [ppm, lorentzian_line, residuals]
for i in range(0, len(popt), 3):
    individual = lorentzian(ppm, popt[i], popt[i+1], popt[i+2])
    columns.append(individual)

result = np.column_stack(columns)

np.savetxt(sys.stdout, result, fmt='%.6f')
