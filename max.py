import numpy as np

data = np.loadtxt("test12_bl.txt")

ppm = data[:, 0]
intensity = data[:, 1]

print(ppm[np.where(intensity == np.max(intensity))])
# for each testXX_bl.txt
# 1. Fit lorentzian lineshapes
# 2. Find max intensity of the curves
# 3. Add this to a list
# 4. Output timecourse_intensity.txt with column format:
#           HOUR    I1  I2  I3  ...
#           3       0.1 0.2 0.1 etc
#           6       0.2 0.2 0.1 etc
