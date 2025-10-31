import pandas as pd
import pandas.api.typing as pat
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=100))
def test_nattype_returns_singleton(n):
    """
    Property: NaTType() should return the same singleton instance as pd.NaT.

    This tests that calling NaTType() multiple times returns the pd.NaT singleton,
    not new instances. This is important because:
    1. pd.NaT is designed as a singleton
    2. pd.isna() and other pandas functions expect the singleton
    3. Identity checks (is) should work
    """
    instances = [pat.NaTType() for _ in range(n)]

    for instance in instances:
        assert instance is pd.NaT, f"NaTType() should return pd.NaT singleton, got different object"
        assert pd.isna(instance), f"pd.isna() should recognize NaTType() instances"

if __name__ == "__main__":
    test_nattype_returns_singleton()