import pandas as pd
import pandas.api.typing as pat
from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=100)
def test_nattype_singleton_property(n):
    instances = [pat.NaTType() for _ in range(n)]

    for i, instance in enumerate(instances):
        assert isinstance(instance, pat.NaTType), f"Instance {i} is not NaTType"
        assert instance is pd.NaT, f"Instance {i} is not the singleton pd.NaT (got {instance} vs {pd.NaT})"

if __name__ == "__main__":
    test_nattype_singleton_property()