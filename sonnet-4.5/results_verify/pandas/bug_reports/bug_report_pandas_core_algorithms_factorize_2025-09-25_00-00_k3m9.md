# Bug Report: pandas.core.algorithms.factorize null character handling

**Target**: `pandas.core.algorithms.factorize`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `factorize` function incorrectly handles strings that contain only null characters (`\x00`), stripping them to empty strings in the returned uniques array, which breaks the round-trip property.

## Property-Based Test

```python
import numpy as np
from pandas.core import algorithms
from hypothesis import given, strategies as st

@given(st.lists(st.text(min_size=0, max_size=20), min_size=1, max_size=50))
def test_factorize_strings_round_trip(values):
    arr = np.array(values)
    codes, uniques = algorithms.factorize(arr)

    assert len(codes) == len(values)

    reconstructed = [uniques[code] if code >= 0 else None for code in codes]
    assert list(reconstructed) == list(values)
```

**Failing input**: `['\x00']`

## Reproducing the Bug

```python
import numpy as np
from pandas.core import algorithms

values = ['\x00']
arr = np.array(values)

codes, uniques = algorithms.factorize(arr)

reconstructed = [uniques[code] for code in codes]

print(f"Original: {repr(values[0])}")
print(f"Reconstructed: {repr(reconstructed[0])}")

assert values[0] == reconstructed[0]
```

Output:
```
Original: '\x00'
Reconstructed: np.str_('')
AssertionError
```

The bug also affects strings like `'\x00\x00'` which becomes `''`, but does not affect strings with null characters in the middle like `'a\x00b'` or other control characters like `'\x01'`.

## Why This Is A Bug

The `factorize` function is documented to encode values as an enumerated type and return codes and uniques such that `uniques[codes[i]] == values[i]`. This round-trip property is violated when the input contains strings that are only null characters. The null characters are stripped, making it impossible to reconstruct the original values from the codes and uniques.

This violates the fundamental contract of the factorize function and could lead to silent data corruption for users working with strings containing null characters.

## Fix

The issue likely stems from how NumPy handles null-terminated strings or how pandas processes the factorized uniques. The fix would need to ensure that null characters are preserved in the uniques array, possibly by using object dtype explicitly or by handling the string encoding differently.