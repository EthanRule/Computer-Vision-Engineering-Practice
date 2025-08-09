import numpy as np
from scipy.stats import norm

pixel_intensities = np.array([12, 18, 25, 30, 42, 45, 45, 50, 55, 60])

mean = np.mean(pixel_intensities)
variance = np.var(pixel_intensities)
standard_deviation = np.std(pixel_intensities)

print(norm.cdf(pixel_intensities, loc=mean, scale=standard_deviation)) #not sure what to do with this