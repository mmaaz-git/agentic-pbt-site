import numpy as np
import matplotlib.pyplot as plt

# Binary search for the threshold where histogram fails
def test_value(val):
    try:
        plt.hist([val])
        plt.close()
        return True
    except ValueError:
        return False

# Test different powers of 10
powers = [10, 12, 13, 14, 15, 16]
for p in powers:
    val = 10.0 ** p
    result = test_value(val)
    print(f"10^{p:2d} ({val:.0e}): {'✓ Works' if result else '✗ Fails'}")

# Find more precise threshold
print("\nFinding precise threshold...")
low = 1e13
high = 1e14
while high - low > 10:
    mid = (low + high) / 2
    if test_value(mid):
        low = mid
    else:
        high = mid

print(f"Threshold is approximately: {low:.2e} to {high:.2e}")