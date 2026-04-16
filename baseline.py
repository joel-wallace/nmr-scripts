import sys
import numpy as np

left_limit = float(sys.argv[1])
right_limit = float(sys.argv[2])

data = np.loadtxt(sys.stdin)
ppm = data[:, 0]
intensity = data[:, 1]

intensity = intensity / (1024*20.9)
degree = 4

mask = (ppm > left_limit) | (ppm < right_limit)

coeffs = np.polyfit(ppm[mask], intensity[mask], degree)

baseline = np.polyval(coeffs, ppm)

corrected = intensity - baseline

result = np.column_stack((ppm, corrected))

np.savetxt(sys.stdout, result, fmt='%.6f %.6f')

