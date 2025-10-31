import numpy as np
import scipy.fftpack as fftpack

x = np.array([-0.5, 0.5])
print(f"Input x: {x}")
print(f"Sum of x: {np.sum(x)}")

diff_x = fftpack.diff(x, order=1)
print(f"diff(x, 1): {diff_x}")

roundtrip = fftpack.diff(diff_x, order=-1)

print(f"diff(diff(x, 1), -1): {roundtrip}")
print(f"Expected: {x}")
print(f"Match: {np.allclose(roundtrip, x)}")

print("\nTesting other examples mentioned in the report:")
print("-" * 50)

test_cases = [
    (np.array([1, -1, 1, -1]), "Length 4: [1, -1, 1, -1]"),
    (np.array([-1, 0, 1]), "Length 3: [-1, 0, 1]"),
    (np.array([0, 1, 0, -1]), "Length 4: [0, 1, 0, -1]"),
]

for x_test, description in test_cases:
    print(f"\n{description}")
    print(f"Sum: {np.sum(x_test)}")
    diff_x_test = fftpack.diff(x_test, order=1)
    roundtrip_test = fftpack.diff(diff_x_test, order=-1)
    print(f"Round-trip successful: {np.allclose(roundtrip_test, x_test)}")
    if not np.allclose(roundtrip_test, x_test):
        print(f"  Original: {x_test}")
        print(f"  Result:   {roundtrip_test}")