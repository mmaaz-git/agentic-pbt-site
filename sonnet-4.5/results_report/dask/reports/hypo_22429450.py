from hypothesis import given, strategies as st
from dask.utils import parse_bytes


@given(
    st.integers(max_value=-1),
    st.sampled_from(['kB', 'MB', 'GB', 'KiB', 'MiB', 'GiB', 'B', ''])
)
def test_parse_bytes_rejects_negative_strings(n, unit):
    s = f"{n}{unit}"
    result = parse_bytes(s)
    assert result >= 0, f"parse_bytes('{s}') returned negative value {result}"

if __name__ == "__main__":
    test_parse_bytes_rejects_negative_strings()