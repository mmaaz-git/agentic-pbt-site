from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import traceback

# First, let's run the hypothesis test
@given(
    values=st.lists(
        st.floats(min_value=-100, max_value=100,
                 allow_nan=False, allow_infinity=False),
        min_size=10, max_size=100
    ),
    n_quantiles=st.integers(min_value=2, max_value=10)
)
@settings(max_examples=100)
def test_qcut_no_crash_on_valid_input(values, n_quantiles):
    series = pd.Series(values).dropna()
    assume(len(series) >= n_quantiles)
    assume(len(series.unique()) >= n_quantiles)

    try:
        result = pd.qcut(series, q=n_quantiles)
        assert len(result) == len(series)
    except ValueError as e:
        if "Bin edges must be unique" in str(e):
            assume(False)
        raise

# Run the hypothesis test with the specific failing input
print("Testing with the specific failing input from the bug report:")
values = [0.0, 1.0, 2.0, 2.2250738585072014e-308]
n_quantiles = 4
series = pd.Series(values).dropna()

print(f"Values: {values}")
print(f"n_quantiles: {n_quantiles}")
print(f"Series length: {len(series)}")
print(f"Unique values: {len(series.unique())}")

try:
    result = pd.qcut(series, q=n_quantiles)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Also test the minimal reproduction code
print("\n" + "="*50)
print("Running the minimal reproduction from the bug report:")
values = [0.0, 1.0, 2.0, 2.2250738585072014e-308]
series = pd.Series(values)

try:
    result = pd.qcut(series, q=4)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()