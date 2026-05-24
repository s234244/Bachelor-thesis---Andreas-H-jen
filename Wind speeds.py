import numpy as np
from py_wake.examples.data.hornsrev1 import Hornsrev1Site

site = Hornsrev1Site()

# Wind speed bins fra Horns Rev 1
ws = site.ds.ws.values
print("Wind speeds [m/s]:")
print(ws)

# Undgå 360 grader, fordi den er samme sektor som 0 grader
wd = site.ds.wd.values[:-1]

# Sandsynligheder for wind direction + wind speed bins
lw = site.local_wind(x=[0], y=[0], wd=wd, ws=ws)
P = lw.P.values

mean_ws = np.sum(ws[None, :] * P) / np.sum(P)

print(f"Mean wind speed: {mean_ws:.3f} m/s")
