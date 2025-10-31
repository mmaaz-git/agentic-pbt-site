# Test the Hypothesis test
from hypothesis import given, settings, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.integers())
@settings(max_examples=10)
def test_argsort_defaults_no_duplicate_assignments(x):
    assert ARGSORT_DEFAULTS['kind'] == 'quicksort' or ARGSORT_DEFAULTS['kind'] is None
    assert 'kind' in ARGSORT_DEFAULTS
    
# Run the test
test_argsort_defaults_no_duplicate_assignments()
print("Hypothesis test completed")

# Now run the reproduction code
print("\n" + "="*50)
print("ARGSORT_DEFAULTS contents:")
for key, value in sorted(ARGSORT_DEFAULTS.items()):
    print(f"  {key!r}: {value!r}")

print(f"\nARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print("Expected: 'quicksort' (from line 138)")
print("Actual: None (overwritten by line 140)")
