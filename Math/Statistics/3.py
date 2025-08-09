import math

mean = 0.4
std = 0.1
x = 0.5


# not sure how to find this porportion

z_score = (x - mean) / std
cdf_z = 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
prob_greater = 1 - cdf_z

print(prob_greater)
print(int(round(prob_greater * 1_000_000)))