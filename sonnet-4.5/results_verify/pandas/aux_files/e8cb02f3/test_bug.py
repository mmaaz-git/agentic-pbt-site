import tokenize
from hypothesis import given, strategies as st
from pandas.core.computation.parsing import clean_column_name

# First, reproduce the direct bug
print("Testing direct reproduction with null byte:")
try:
    result = clean_column_name('\x00')
    print(f"Result: {result}")
except tokenize.TokenError as e:
    print(f"TokenError raised: {e}")
except SyntaxError as e:
    print(f"SyntaxError raised: {e}")

# Now test with hypothesis
@given(st.text())
def test_clean_column_name_no_crash(name):
    try:
        result = clean_column_name(name)
        assert isinstance(result, type(name))
    except SyntaxError:
        pass
    except tokenize.TokenError:
        raise AssertionError(f"TokenError not caught for input: {name!r}")

print("\nRunning hypothesis test:")
try:
    test_clean_column_name_no_crash()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")