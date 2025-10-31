import sys
from hypothesis import assume, given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utils import normalise_float_repr


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50))
def test_normalise_float_repr_preserves_value(f):
    assume(f != 0.0 or str(f) != '-0.0')

    float_str = str(f)
    normalised = normalise_float_repr(float_str)

    original_value = float(float_str)
    normalised_value = float(normalised)

    assert original_value == normalised_value

if __name__ == "__main__":
    test_normalise_float_repr_preserves_value()