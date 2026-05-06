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
def residuals_function(params, x, y):
    current_fit = sum_lorentzians(params, x)
    return current_fit - y

# user guesses
p0 = jnp.array([0.08, -61.85, 0.05, 0.03, -62.67, 0.05, 0.022, -61.85, 0.3, 0.022, -62.67, 0.3])

solver = jaxopt.LevenbergMarquardt(residual_fun=residuals_function)

# initial fit
result = solver.run(p0, x=ppm, y=intensity)
# popt stands for optimized parameters
popt = result.params
# draws optimised summed line
lorentzian_line = sum_lorentzians(popt, ppm)
residuals = intensity - lorentzian_line

spectrometer_freq = 470.611

# 1. Calculate Area (Integral) for the main fit: A * |gamma| * pi
integrals = jnp.pi * popt[0::3] * jnp.abs(popt[2::3])

# 2. Convert gamma (HWHM) to Hz for the main fit
lw_hz = popt[2::3] * spectrometer_freq

# Calculate relative fractions for the main fit
total_integral = jnp.sum(integrals)
rel_integrals = integrals / total_integral

# --- PRINT TERMINAL SUMMARY ---
sys.stderr.write("\n--- RESULTS ---\n")
# Updated table headers for the new parameters (expanded width for new column)
sys.stderr.write(f"{'Peak':<5} | {'Intensity (A)':<15} | {'Shift (ppm)':<15} | {'lw (Hz)':<12} | {'Abs Integral':<15} | {'Rel Integral':<15}\n")
sys.stderr.write("-" * 105 + "\n")

# Loop per peak rather than per flat parameter
for i in range(int(len(popt) / 3)):
    peak_idx = i + 1

    # Extract raw values and errors for this specific peak
    A_val = popt[i*3]
    shift_val = popt[i*3 + 1]
    
    # Extract derived values and errors
    lw_hz_val = lw_hz[i]
    int_val = integrals[i]
    rel_int_val = rel_integrals[i]

    sys.stderr.write(f"{peak_idx:<5} | {A_val:<15.6f} | {shift_val:<15.6f} | {lw_hz_val:<12.6f} | {int_val:<15.6f} | {rel_int_val:<15.6f}\n")
    sys.stderr.write("-" * 105 + "\n")
sys.stderr.write("\n")

# columns for output file
columns = [ppm, lorentzian_line, residuals]
# draws the individual lines
for i in range(0, len(popt), 3):
    individual = lorentzian(ppm, popt[i], popt[i+1], popt[i+2])
    columns.append(individual)


# output
result = np.column_stack(columns)
np.savetxt(sys.stdout, result, fmt='%.6f')
