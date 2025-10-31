from hypothesis import given, strategies as st
from pandas.compat._optional import get_version
import types

@given(
    version=st.text(min_size=1, max_size=20).filter(lambda x: ' ' not in x)
)
def test_get_version_psycopg2_special_case(version):
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = f"{version} (dt dec pq3 ext lo64)"

    result = get_version(mock_module)
    assert result == version

# Run the test
if __name__ == "__main__":
    test_get_version_psycopg2_special_case()