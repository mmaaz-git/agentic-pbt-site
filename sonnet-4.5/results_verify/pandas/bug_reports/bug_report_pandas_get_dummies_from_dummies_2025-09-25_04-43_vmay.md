# Bug Report: pandas get_dummies/from_dummies Empty DataFrame Round-Trip

**Target**: `pandas.get_dummies` and `pandas.from_dummies`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The round-trip property `from_dummies(get_dummies(df)) == df` fails for empty DataFrames, losing column information.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings

@settings(max_examples=500)
@given(
    data=st.lists(
        st.tuples(
            st.sampled_from(['a', 'b', 'c', 'd']),
            st.sampled_from(['x', 'y', 'z'])
        ),
        max_size=50
    )
)
def test_get_dummies_from_dummies_round_trip(data):
    df = pd.DataFrame(data, columns=['A', 'B'])
    dummies = pd.get_dummies(df, dtype=int)
    recovered = pd.from_dummies(dummies, sep='_')

    pd.testing.assert_frame_equal(df, recovered)
```

**Failing input**: `data=[]`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame(columns=['A', 'B'])
print('Original:', df.shape, list(df.columns))

dummies = pd.get_dummies(df, dtype=int)
print('After get_dummies:', dummies.shape, list(dummies.columns))

recovered = pd.from_dummies(dummies, sep='_')
print('After from_dummies:', recovered.shape, list(recovered.columns))

print('Equal?', df.equals(recovered))
```

Output:
```
Original: (0, 2) ['A', 'B']
After get_dummies: (0, 0) []
After from_dummies: (0, 0) []
Equal? False
```

## Why This Is A Bug

The documentation for `from_dummies` states it "inverts the operation performed by `get_dummies`". However, for empty DataFrames, the round-trip loses all column information. This violates the inverse operation contract and would cause issues in data pipelines that process DataFrames that might be empty.

## Fix

The issue is that `get_dummies` on an empty DataFrame produces an empty DataFrame with no columns, losing information about which columns existed. When `from_dummies` processes this, it has no way to recover the original column structure.

A potential fix would be to make `get_dummies` preserve column information even for empty DataFrames by creating dummy columns with the appropriate naming scheme, or to document this edge case as a known limitation. The fix likely needs to be in `get_dummies` rather than `from_dummies`, as the column information is lost before `from_dummies` is called.