from hypothesis import given
from hypothesis import strategies as st
from xarray.core.dtypes import AlwaysGreaterThan

@given(st.integers())
def test_alwaysgt_comparison_properties(x):
    agt = AlwaysGreaterThan()

    if not isinstance(x, AlwaysGreaterThan):
        assert agt > x

    agt2 = AlwaysGreaterThan()
    assert agt == agt2
    assert not (agt > agt2)

if __name__ == "__main__":
    test_alwaysgt_comparison_properties()