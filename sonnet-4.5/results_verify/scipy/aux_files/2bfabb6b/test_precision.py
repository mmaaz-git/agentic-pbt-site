from hypothesis import given, strategies as st
import scipy.constants

@given(st.sampled_from(list(scipy.constants.physical_constants.keys())))
def test_precision_returns_relative_not_absolute(key):
    val, unit, uncertainty = scipy.constants.physical_constants[key]
    result = scipy.constants.precision(key)

    if val != 0 and uncertainty != 0:
        expected_relative = uncertainty / val
        assert result == expected_relative, f"Expected {expected_relative}, got {result}"

# Run the test
if __name__ == "__main__":
    # Test with the specific example from the docstring
    key = 'proton mass'
    val, unit, uncertainty = scipy.constants.physical_constants[key]
    result = scipy.constants.precision(key)

    print(f"Testing key: {key}")
    print(f"Value: {val}")
    print(f"Unit: {unit}")
    print(f"Uncertainty: {uncertainty}")
    print(f"Result from precision(): {result}")
    print(f"Expected (uncertainty/value): {uncertainty/val}")
    print(f"Match: {result == uncertainty/val}")

    # Run the hypothesis test
    test_precision_returns_relative_not_absolute()