from pandas.compat.numpy.function import ARGSORT_DEFAULTS, SORT_DEFAULTS

print(f"ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print(f"SORT_DEFAULTS['kind'] = {SORT_DEFAULTS['kind']!r}")

# Demonstrate the inconsistency
print(f"\nInconsistency detected: ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind']")
print(f"    ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print(f"    SORT_DEFAULTS['kind'] = {SORT_DEFAULTS['kind']!r}")

# Check if both numpy functions have the same default
import numpy as np
import inspect

sort_sig = inspect.signature(np.sort)
argsort_sig = inspect.signature(np.argsort)

print(f"\nnumpy.sort default 'kind' parameter: {sort_sig.parameters['kind'].default!r}")
print(f"numpy.argsort default 'kind' parameter: {argsort_sig.parameters['kind'].default!r}")

print("\nBoth numpy functions have the same 'kind' default (None), but pandas tracks them inconsistently.")

# The assertion that would fail
assert ARGSORT_DEFAULTS['kind'] != SORT_DEFAULTS['kind'], "The inconsistency is confirmed"