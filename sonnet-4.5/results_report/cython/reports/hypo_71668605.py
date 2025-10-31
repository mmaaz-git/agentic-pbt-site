from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Utility.Dataclasses import field

@given(st.booleans())
def test_field_repr_consistency(kw_only_value):
    f = field(kw_only=kw_only_value)
    repr_str = repr(f)
    # The repr should use the same attribute name as the actual attribute (kw_only, not kwonly)
    assert f'kw_only={kw_only_value!r}' in repr_str, \
        f"Expected 'kw_only={kw_only_value!r}' in repr, but got 'kwonly={kw_only_value!r}' instead. Full repr: {repr_str}"

if __name__ == "__main__":
    test_field_repr_consistency()