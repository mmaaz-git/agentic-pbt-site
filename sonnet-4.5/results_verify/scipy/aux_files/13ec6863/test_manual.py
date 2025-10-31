import numpy as np
import scipy.signal.windows as windows

test_cases = [
    (2, 1e-308),
    (5, 1e-308),
    (10, 1e-308),
]

print("Testing manual test cases from bug report:")
print("=" * 50)
for M, alpha in test_cases:
    w = windows.tukey(M, alpha)
    print(f"tukey({M}, alpha={alpha:.2e}) = {w}")