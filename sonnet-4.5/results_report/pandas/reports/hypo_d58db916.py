from hypothesis import given, strategies as st, settings
import pandas.api.typing as typing
import pandas as pd


@given(st.integers(min_value=0, max_value=100))
@settings(max_examples=100)
def test_nattype_singleton_property(n):
    instances = [typing.NaTType() for _ in range(n)]
    if len(instances) > 0:
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, f"NaTType() should always return the same singleton instance"
            assert instance is pd.NaT, f"NaTType() should return pd.NaT"

if __name__ == "__main__":
    test_nattype_singleton_property()