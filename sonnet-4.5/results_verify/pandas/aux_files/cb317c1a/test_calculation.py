import numpy as np
import math

# Simulate the calculation for M=2, alpha=1e-300
M = 2
alpha = 1e-300

print(f"M = {M}, alpha = {alpha}")

# Calculate width
width = int(math.floor(alpha*(M-1)/2.0))
print(f"width = {width}")

# Create indices
n = np.arange(0, M, dtype=np.float64)
print(f"n = {n}")

n1 = n[0:width+1]
n2 = n[width+1:M-width-1]
n3 = n[M-width-1:]

print(f"n1 = {n1}")
print(f"n2 = {n2}")
print(f"n3 = {n3}")

# Calculate w1
if len(n1) > 0:
    arg1 = -1 + 2.0*n1/alpha/(M-1)
    print(f"arg1 for w1 = {arg1}")
    w1 = 0.5 * (1 + np.cos(np.pi * arg1))
    print(f"w1 = {w1}")
else:
    w1 = np.array([])
    print("w1 is empty")

# Calculate w2
w2 = np.ones(n2.shape)
print(f"w2 = {w2}")

# Calculate w3 - the problematic calculation
if len(n3) > 0:
    print("\nDetailed calculation for w3:")
    print(f"n3 = {n3}")

    # Original (problematic) calculation
    term1 = -2.0/alpha
    term2 = 1
    term3 = 2.0*n3/alpha/(M-1)

    print(f"  term1 (-2.0/alpha) = {term1}")
    print(f"  term2 (1) = {term2}")
    print(f"  term3 (2.0*n3/alpha/(M-1)) = {term3}")

    # Left-to-right evaluation (what Python does)
    step1 = term1 + term2
    print(f"  step1 (term1 + term2) = {step1}")
    step2 = step1 + term3
    print(f"  step2 (step1 + term3) = {step2}")

    arg3_wrong = step2
    print(f"  arg3 (wrong) = {arg3_wrong}")
    w3_wrong = 0.5 * (1 + np.cos(np.pi * arg3_wrong))
    print(f"  w3 (wrong) = {w3_wrong}")

    # Correct calculation (grouping large terms)
    arg3_correct = 1 + (term1 + term3)
    print(f"  arg3 (correct, grouping large terms) = {arg3_correct}")
    w3_correct = 0.5 * (1 + np.cos(np.pi * arg3_correct))
    print(f"  w3 (correct) = {w3_correct}")
else:
    w3 = np.array([])
    print("w3 is empty")

# Full window
w_wrong = np.concatenate((w1, w2, w3_wrong))
w_correct = np.concatenate((w1, w2, w3_correct))

print(f"\nFinal window (wrong): {w_wrong}")
print(f"Final window (correct): {w_correct}")
print(f"Wrong is symmetric: {np.allclose(w_wrong, w_wrong[::-1])}")
print(f"Correct is symmetric: {np.allclose(w_correct, w_correct[::-1])}")