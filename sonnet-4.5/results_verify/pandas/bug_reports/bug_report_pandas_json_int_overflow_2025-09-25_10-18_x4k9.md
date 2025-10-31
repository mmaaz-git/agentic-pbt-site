# Bug Report: pandas JSON Round-Trip Fails for Integers Below int64.min

**Target**: `pandas.DataFrame.to_json()` and `pandas.read_json()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

DataFrames containing integers smaller than int64.min can be serialized with `to_json()` but cannot be deserialized with `read_json()`, violating the fundamental round-trip property.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st

@given(st.data())
def test_json_round_trip_int_overflow(data):
    n = data.draw(st.integers(min_value=1, max_value=20))
    df = pd.DataFrame({
        'a': data.draw(st.lists(st.integers(), min_size=n, max_size=n)),
        'b': data.draw(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    })

    json_str = df.to_json()
    reconstructed = pd.read_json(json_str)

    assert len(df) == len(reconstructed)
```

**Failing input**: `{'a': [-9223372036854775809], 'b': [0.0]}`

## Reproducing the Bug

```python
import pandas as pd

df = pd.DataFrame({'a': [-9223372036854775809]})
json_str = df.to_json()
reconstructed = pd.read_json(json_str)
```

**Output**:
```
ValueError: Value is too small
```

## Why This Is A Bug

The value `-9223372036854775809` is one less than `np.iinfo(np.int64).min` (-9223372036854775808). While Python supports arbitrarily large integers, pandas stores such values as object dtype. The bug manifests because:

1. `DataFrame.to_json()` successfully serializes this value to JSON: `{"a":{"0":-9223372036854775809}}`
2. `read_json()` fails to parse it back, raising "ValueError: Value is too small"

This violates the expected serialization contract: if `to_json()` produces output, `read_json()` should be able to parse it back. The asymmetry is notable: integers above int64.max (stored as uint64) successfully round-trip, but integers below int64.min do not.

## Fix

The underlying issue is in the ujson parser used by `read_json()` which cannot handle integers below int64.min. Pandas should either:

1. Validate during `to_json()` and raise an error for unparseable values
2. Convert such values to strings during serialization to preserve data
3. Use a different JSON parser that supports arbitrary precision integers

A simple validation fix in the `to_json()` method would check for out-of-range integers in object-dtype columns before serialization and either raise an error or convert them to strings.