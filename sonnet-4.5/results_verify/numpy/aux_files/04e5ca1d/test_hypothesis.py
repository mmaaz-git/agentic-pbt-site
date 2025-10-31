import numpy as np
import numpy.rec
from hypothesis import given, strategies as st, settings
import pytest

RECARRAY_METHOD_NAMES = [
    'field', 'item', 'copy', 'view', 'tolist', 'fill', 'all', 'any',
    'argmax', 'argmin', 'argsort', 'astype', 'byteswap', 'choose', 'clip',
    'compress', 'conj', 'conjugate', 'cumprod', 'cumsum', 'diagonal', 'dot',
    'dump', 'dumps', 'flatten', 'getfield', 'max', 'mean', 'min', 'nonzero',
    'partition', 'prod', 'ptp', 'put', 'ravel', 'repeat', 'reshape', 'resize',
    'round', 'searchsorted', 'setfield', 'setflags', 'sort', 'squeeze', 'std',
    'sum', 'swapaxes', 'take', 'tobytes', 'trace', 'transpose', 'var'
]

@given(st.sampled_from(RECARRAY_METHOD_NAMES),
       st.lists(st.integers(), min_size=1, max_size=10))
@settings(max_examples=10)  # Reduce number of examples for faster testing
def test_method_name_fields_accessible_via_attribute(method_name, data):
    rec = numpy.rec.fromrecords([(x,) for x in data], names=method_name)

    dict_access = rec[method_name]
    attr_access = getattr(rec, method_name)

    if isinstance(attr_access, np.ndarray):
        np.testing.assert_array_equal(attr_access, dict_access)
        print(f"✓ Field '{method_name}' works correctly")
    else:
        print(f"✗ Field '{method_name}': rec.{method_name} returned {type(attr_access).__name__} instead of field data.")
        print(f"  Dictionary access rec['{method_name}'] works correctly, but attribute access rec.{method_name} returns a method.")
        pytest.fail(f"Field '{method_name}': rec.{method_name} returned {type(attr_access).__name__} instead of field data. "
                   f"Dictionary access rec['{method_name}'] works correctly, but attribute access rec.{method_name} returns a method.")

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test for recarray field name conflicts...")
    try:
        test_method_name_fields_accessible_via_attribute()
    except Exception as e:
        print(f"\nTest failed as expected: {str(e)[:200]}...")
    else:
        print("\nTest passed (should not happen if bug exists)")