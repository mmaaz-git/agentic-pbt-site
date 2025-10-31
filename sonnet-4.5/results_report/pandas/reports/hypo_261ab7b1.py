import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.core.window.common import prep_binary


@given(st.lists(st.floats(allow_nan=False, allow_infinity=True), min_size=1, max_size=50))
@settings(max_examples=1000)
def test_prep_binary_infinity_handling(values):
    has_inf = any(np.isinf(v) for v in values)
    if not has_inf:
        return

    s1_values = [1.0] * len(values)
    s1 = pd.Series(s1_values)
    s2 = pd.Series(values)

    X, Y = prep_binary(s1, s2)

    for i in range(len(values)):
        if np.isinf(s2.iloc[i]) and not np.isnan(s1.iloc[i]):
            assert not np.isnan(X.iloc[i]), \
                f"Finite value s1[{i}]={s1.iloc[i]} became NaN due to inf in s2[{i}]"

if __name__ == "__main__":
    test_prep_binary_infinity_handling()