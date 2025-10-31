from hypothesis import given, strategies as st, settings
from Cython.Build.Dependencies import extended_iglob
import warnings


@given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/*', min_size=5, max_size=50))
@settings(max_examples=500)
def test_extended_iglob_no_warnings(pattern):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        list(extended_iglob(pattern))
        assert len(w) == 0, f"extended_iglob should not generate warnings: {[str(x.message) for x in w]}"

if __name__ == "__main__":
    test_extended_iglob_no_warnings()