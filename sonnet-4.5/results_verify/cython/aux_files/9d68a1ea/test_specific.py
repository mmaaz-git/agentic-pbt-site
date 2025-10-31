from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import extended_iglob
import warnings

# Test with specific pattern that triggers the issue
pattern = '**/*.py'
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    list(extended_iglob(pattern))
    if len(w) > 0:
        print(f"Test failed with pattern '{pattern}'")
        for warning in w:
            print(f"  {warning.category.__name__}: {warning.message}")
    else:
        print(f"No warnings for pattern '{pattern}'")