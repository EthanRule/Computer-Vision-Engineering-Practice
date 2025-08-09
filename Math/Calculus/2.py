# L(p) = (p - 1)^2
p = 5.0
t = 3.0

# dL/dp = 2(p - t)

def compute(p, t):
    dLdp = 2 * (p - t)
    Lp = (p - t)**2
    return (dLdp, Lp)

print(compute(p, t))