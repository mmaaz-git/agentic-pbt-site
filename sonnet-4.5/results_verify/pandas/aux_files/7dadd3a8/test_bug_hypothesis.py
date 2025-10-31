from hypothesis import given, strategies as st
import pytest
from pandas.compat.numpy.function import validate_argsort_with_ascending


@given(ascending=st.booleans(),
       kind=st.one_of(st.none(), st.just('quicksort'), st.just('mergesort')))
def test_validate_argsort_kind_accepts_kind_parameter(ascending, kind):
    kwargs = {"kind": kind} if kind is not None else {}

    try:
        result = validate_argsort_with_ascending(ascending, (), kwargs)
        assert isinstance(result, bool)
    except TypeError as e:
        if "unexpected keyword argument 'kind'" in str(e):
            pytest.fail(f"validate_argsort_kind should accept 'kind' parameter but got: {e}")
        raise

if __name__ == "__main__":
    # Run the test with specific failing input as mentioned in the bug report
    ascending = True
    kwargs = {"kind": None}  # Explicitly pass kind=None

    try:
        result = validate_argsort_with_ascending(ascending, (), kwargs)
        print(f"Test passed with ascending={ascending}, kwargs={kwargs}, result={result}")
    except TypeError as e:
        if "unexpected keyword argument 'kind'" in str(e):
            print(f"Bug confirmed: validate_argsort_kind should accept 'kind' parameter but got: {e}")
        else:
            raise