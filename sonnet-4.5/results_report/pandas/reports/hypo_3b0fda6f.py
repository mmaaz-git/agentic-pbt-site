from hypothesis import given, strategies as st, settings, Phase
from pandas.io.parsers.readers import _validate_names
import math

@given(st.lists(st.floats(allow_nan=True), min_size=2, max_size=10))
@settings(max_examples=100, phases=[Phase.generate, Phase.target])
def test_validate_names_detects_nan_duplicates(names):
    nan_count = sum(1 for x in names if isinstance(x, float) and math.isnan(x))
    if nan_count > 1:
        try:
            _validate_names(names)
            assert False, f"Should reject duplicate NaN in {names}"
        except ValueError:
            pass

if __name__ == "__main__":
    test_validate_names_detects_nan_duplicates()