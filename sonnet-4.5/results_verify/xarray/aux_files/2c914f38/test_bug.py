from hypothesis import given, strategies as st
from xarray.core.formatting_html import _load_static_files


@given(st.text())
def test_cache_returns_independent_copies(mutation_value):
    call1 = _load_static_files()
    call2 = _load_static_files()

    call1.append(mutation_value)

    assert len(call2) == 2


# Run the test
if __name__ == "__main__":
    test_cache_returns_independent_copies()