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
    
    def fit_single(y_sample):
        return solver.run(ipopt, bounds=bounds, x=x, y=y_sample).params

    results = jax.lax.map(fit_single, sample_y)
    return results

test = bootstrap(ppm, lorentzian_line, 200, popt, residuals, bounds)

std_errors = jnp.std(test, axis=0)

spectrometer_freq = 470.611

# 1. Calculate Area (Integral) for the main fit: A * |gamma| * pi
integrals = jnp.pi * popt[0::3] * jnp.abs(popt[2::3])

# 2. Convert gamma (HWHM) to Hz for the main fit
lw_hz = popt[2::3] * spectrometer_freq

# 3. Calculate exact standard errors for derived values using the bootstrap matrix
boot_A = test[:, 0::3]
boot_gamma = test[:, 2::3]

boot_integrals = jnp.pi * boot_A * jnp.abs(boot_gamma)
integral_errors = jnp.std(boot_integrals, axis=0)

lw_hz_errors = jnp.std(boot_gamma * spectrometer_freq, axis=0)

# Calculate relative fractions for the main fit
total_integral = jnp.sum(integrals)
rel_integrals = integrals / total_integral

# Calculate exact standard errors for relative integrals using the bootstrap matrix
# axis=1 sums across the peaks for each of the 1000 rows
boot_total_integrals = jnp.sum(boot_integrals, axis=1, keepdims=True)
boot_rel_integrals = boot_integrals / boot_total_integrals
rel_integral_errors = jnp.std(boot_rel_integrals, axis=0)

# --- PRINT TERMINAL SUMMARY ---
sys.stderr.write("\n--- RESULTS ---\n")
# Updated table headers for the new parameters (expanded width for new column)
sys.stderr.write(f"{'Peak':<5} | {'Intensity (A)':<15} | {'Shift (ppm)':<15} | {'lw (Hz)':<12} | {'Abs Integral':<15} | {'Rel Integral':<15}\n")
sys.stderr.write("-" * 105 + "\n")

# Loop per peak rather than per flat parameter
for i in range(int(len(popt) / 3)):
    peak_idx = i + 1

    # Extract raw values and errors for this specific peak
    A_val, A_err = popt[i*3], std_errors[i*3]
    shift_val, shift_err = popt[i*3 + 1], std_errors[i*3 + 1]
    
    # Extract derived values and errors
    lw_hz_val, lw_hz_err_val = lw_hz[i], lw_hz_errors[i]
    int_val, int_err = integrals[i], integral_errors[i]
    rel_int_val, rel_int_err = rel_integrals[i], rel_integral_errors[i]

    sys.stderr.write(f"{peak_idx:<5} | {A_val:<15.6f} | {shift_val:<15.6f} | {lw_hz_val:<12.6f} | {int_val:<15.6f} | {rel_int_val:<15.6f}\n")
    sys.stderr.write(f"{'':<5} | ±{A_err:<14.6f} | ±{shift_err:<14.6f} | ±{lw_hz_err_val:<11.6f} | ±{int_err:<14.6f} | ±{rel_int_err:<14.6f}\n")
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
