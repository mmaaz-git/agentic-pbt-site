import numpy as np
import scipy.special

print("Testing round-trip property: logit(expit(x)) should equal x")
print("="*60)

print("\nFor large NEGATIVE x (works perfectly):")
for x in [-10, -20, -30, -40, -50, -100]:
    result = scipy.special.logit(scipy.special.expit(x))
    error = abs(result - x)
    print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")

print("\nFor large POSITIVE x (catastrophic failure):")
for x in [10, 20, 30, 40, 50, 100]:
    result = scipy.special.logit(scipy.special.expit(x))
    if np.isfinite(result):
        error = abs(result - x)
        print(f"  logit(expit({x:4})) = {result:8.2f}, error = {error:.2e}")
    else:
        print(f"  logit(expit({x:4})) = {result:>8}, error = inf")

print("\n" + "="*60)
print("Demonstrating the root cause - catastrophic cancellation:")
print()

x = 20.0
p = scipy.special.expit(x)
print(f"x = {x}")
print(f"p = expit({x}) = {p:.20f}")
print(f"  (note: p is very close to 1.0)")

one_minus_p_naive = 1 - p
one_minus_p_accurate = np.exp(-x) / (1 + np.exp(-x))

print(f"\n1-p computed naively (1 - {p:.20f}):")
print(f"  1-p = {one_minus_p_naive:.20e}")

print(f"\n1-p computed accurately (exp(-x)/(1+exp(-x))):")
print(f"  1-p = {one_minus_p_accurate:.20e}")

logit_naive = np.log(p / one_minus_p_naive)
logit_accurate = np.log(p / one_minus_p_accurate)

print(f"\nlogit(p) using naive 1-p:")
print(f"  Result: {logit_naive:.10f}, error = {abs(logit_naive - x):.2e}")

print(f"\nlogit(p) using accurate 1-p:")
print(f"  Result: {logit_accurate:.10f}, error = {abs(logit_accurate - x):.2e}")

print("\n" + "="*60)
print("Testing critical value where it becomes infinity:")
print()
for x in [35, 36, 37, 38]:
    p = scipy.special.expit(x)
    result = scipy.special.logit(p)
    print(f"x = {x}: expit({x}) = {p:.20f}")
    print(f"         logit(expit({x})) = {result}")
    print()