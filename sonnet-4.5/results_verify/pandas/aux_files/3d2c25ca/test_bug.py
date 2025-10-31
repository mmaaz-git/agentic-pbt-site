import inspect
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from pandas.compat.numpy.function import validate_sort, SORT_DEFAULTS


def test_sort_defaults_should_match_numpy():
    """
    Property: SORT_DEFAULTS should contain numpy.sort's actual default values
    so that validation correctly accepts numpy defaults.
    """
    sig = inspect.signature(np.sort)
    numpy_kind_default = sig.parameters['kind'].default

    assert SORT_DEFAULTS['kind'] == numpy_kind_default, \
        f"SORT_DEFAULTS['kind']={SORT_DEFAULTS['kind']!r} " \
        f"but numpy default is {numpy_kind_default!r}"


def test_validate_sort_should_accept_numpy_default():
    """
    Property: validate_sort should accept numpy.sort's actual default value.
    """
    validate_sort((), {'kind': None})

# Basic reproduction test
print("Basic reproduction:")
sig = inspect.signature(np.sort)
numpy_default = sig.parameters['kind'].default

print(f"numpy.sort default for 'kind': {numpy_default!r}")
print(f"SORT_DEFAULTS['kind']: {SORT_DEFAULTS['kind']!r}")

try:
    validate_sort((), {'kind': None})
    print("validate_sort with kind=None: SUCCESS")
except Exception as e:
    print(f"validate_sort with kind=None: FAILED with {type(e).__name__}: {e}")

try:
    validate_sort((), {'kind': 'quicksort'})
    print("validate_sort with kind='quicksort': SUCCESS")
except Exception as e:
    print(f"validate_sort with kind='quicksort': FAILED with {type(e).__name__}: {e}")

# Now run the tests
print("\nRunning hypothesis tests...")
try:
    test_sort_defaults_should_match_numpy()
    print("test_sort_defaults_should_match_numpy: PASSED")
except AssertionError as e:
    print(f"test_sort_defaults_should_match_numpy: FAILED - {e}")

try:
    test_validate_sort_should_accept_numpy_default()
    print("test_validate_sort_should_accept_numpy_default: PASSED")
except Exception as e:
    print(f"test_validate_sort_should_accept_numpy_default: FAILED - {type(e).__name__}: {e}")