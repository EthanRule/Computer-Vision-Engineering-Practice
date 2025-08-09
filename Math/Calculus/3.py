# L(p) = (p - t)^2 + 0.1*p^4

# partial = 2p + 4p^3

p = 0.6
t = 0.8
def gradient(p, t):
    return 2 * (p - t) + 0.4 * p**3

print(gradient(0.6, 0.8))