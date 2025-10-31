"""Minimal reproduction of the ARGSORT_DEFAULTS duplicate key bug in pandas"""

# Import the module that contains the bug
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, ARGSORT_DEFAULTS_KIND

print("=== Demonstrating the ARGSORT_DEFAULTS duplicate key bug ===")
print()
print("Contents of ARGSORT_DEFAULTS dictionary:")
for key, value in ARGSORT_DEFAULTS.items():
    print(f"  {key}: {value!r}")
print()

print("Value of ARGSORT_DEFAULTS['kind']:", ARGSORT_DEFAULTS["kind"])
print()
print("Expected behavior: Should be either 'quicksort' OR None, not both assigned")
print("Actual behavior: The value is None (line 140 overwrites line 138)")
print()

print("For comparison, ARGSORT_DEFAULTS_KIND dictionary (which omits 'kind'):")
for key, value in ARGSORT_DEFAULTS_KIND.items():
    print(f"  {key}: {value!r}")
print()

print("Note: ARGSORT_DEFAULTS_KIND intentionally omits 'kind' key")
print("This suggests ARGSORT_DEFAULTS should have 'kind' set to 'quicksort'")