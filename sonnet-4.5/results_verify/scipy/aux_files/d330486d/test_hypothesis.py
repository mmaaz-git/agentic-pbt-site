from hypothesis import given, strategies as st, settings, assume
from scipy.optimize.cython_optimize import _zeros

@given(
    st.floats(min_value=-5.0, max_value=-0.1, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_no_sign_change_should_error(a0, offset):
    true_root = (-a0) ** (1.0/3.0)

    xa = true_root + offset
    xb = true_root + offset + 1.0

    f_xa = xa**3 + a0
    f_xb = xb**3 + a0

    assume(f_xa > 0 and f_xb > 0)

    methods = ['bisect', 'brentq', 'brenth', 'ridder']

    for method in methods:
        results = list(_zeros.loop_example(method, (a0,), (0.0, 0.0, 1.0), xa, xb, 0.01, 0.01, 50))

        assert len(results) == 0 or results[0] != 0.0, \
            f"{method} should error or return valid result when no sign change"

# Run the test
print("Running hypothesis test...")
try:
    test_no_sign_change_should_error()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Also test the specific case from the bug report
print("\nTesting specific case from bug report:")
a0 = -1.0
offset = 1.0
true_root = (-a0) ** (1.0/3.0)
xa = true_root + offset
xb = true_root + offset + 1.0

print(f"a0={a0}, offset={offset}")
print(f"true_root={true_root}")
print(f"xa={xa}, xb={xb}")

def f(x):
    return x**3 + a0

f_xa = f(xa)
f_xb = f(xb)

print(f"f(xa)={f_xa}, f(xb)={f_xb}")
print(f"Both values are positive: {f_xa > 0 and f_xb > 0}")

print("\nTesting each method:")
for method in ['bisect', 'brentq', 'brenth', 'ridder']:
    results = list(_zeros.loop_example(method, (a0,), (0.0, 0.0, 1.0), xa, xb, 0.01, 0.01, 50))
    print(f"  {method}: results={results}")
    if results and results[0] == 0.0:
        print(f"    FAIL: Returned 0.0 which is invalid")