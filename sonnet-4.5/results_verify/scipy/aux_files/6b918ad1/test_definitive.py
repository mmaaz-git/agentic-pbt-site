import numpy as np
from scipy.interpolate import PPoly

print("DEFINITIVE TEST: Continuous vs Discontinuous PPoly")
print("="*60)

# Case 1: Discontinuous piecewise constant
print("\n1. DISCONTINUOUS piecewise constant:")
print("-" * 40)
c_disc = np.array([[1.0, 2.0]])  # Constants: 1 on [0,1], 2 on [1,2]
x = np.array([0.0, 1.0, 2.0])
p_disc = PPoly(c_disc, x)

p_disc_deriv = p_disc.derivative()
p_disc_anti = p_disc_deriv.antiderivative()

x_test = np.array([0.5, 1.5])
for xi in x_test:
    orig = p_disc(xi)
    recon = p_disc_anti(xi)
    diff = orig - recon
    print(f"  x={xi}: p(x)={orig:.1f}, p'.anti(x)={recon:.1f}, diff={diff:.1f}")

diffs_disc = p_disc(x_test) - p_disc_anti(x_test)
is_const_disc = np.allclose(diffs_disc[0], diffs_disc[1])
print(f"  Difference is constant? {is_const_disc}")

# Case 2: Continuous piecewise polynomial (manually constructed)
print("\n2. CONTINUOUS piecewise polynomial:")
print("-" * 40)

# Create continuous quadratic/linear combination
# Piece 1: p1(x) = x on [0,1], so p1(1) = 1
# Piece 2: p2(x) = 2*x - 1 on [1,2], so p2(1) = 1 (continuous!)
# But wait, PPoly uses local coordinates!

# In PPoly, each piece uses local coordinates from 0 to h_i
# So we need to be careful with the coefficients

# Actually, let's use a simpler approach with BSpline conversion
from scipy.interpolate import BSpline

# Create a simple B-spline (always continuous)
t = [0, 0, 1, 2, 2]  # knot vector for linear spline
c = [0, 1, 0]  # control points
k = 1  # linear spline
bspl = BSpline(t, c, k)

# Convert to PPoly
p_cont = PPoly.from_spline(bspl)

p_cont_deriv = p_cont.derivative()
p_cont_anti = p_cont_deriv.antiderivative()

x_test = np.array([0.5, 1.5])
for xi in x_test:
    orig = p_cont(xi)
    recon = p_cont_anti(xi)
    diff = orig - recon
    print(f"  x={xi}: p(x)={orig:.4f}, p'.anti(x)={recon:.4f}, diff={diff:.4f}")

diffs_cont = p_cont(x_test) - p_cont_anti(x_test)
is_const_cont = np.allclose(diffs_cont[0], diffs_cont[1], atol=1e-8)
print(f"  Difference is constant? {is_const_cont}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print(f"Discontinuous PPoly: derivative().antiderivative() differs by")
print(f"  {'CONSTANT' if is_const_disc else 'NON-CONSTANT'} amount")
print(f"Continuous PPoly: derivative().antiderivative() differs by")
print(f"  {'CONSTANT' if is_const_cont else 'NON-CONSTANT'} amount")

if not is_const_disc and is_const_cont:
    print("\n→ The docstring claim holds for continuous polynomials only!")
    print("  For discontinuous ones, antiderivative() enforces continuity,")
    print("  breaking the inverse relationship.")