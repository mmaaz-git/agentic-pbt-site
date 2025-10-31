# Bug Report: scipy.sparse COO Canonical Format Flag Not Set on CSR Conversion

**Target**: `scipy.sparse.coo_matrix`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When converting a CSR matrix to COO format using `tocoo()`, the resulting COO matrix has no duplicate entries (as expected from CSR format) but the `has_canonical_format` flag is incorrectly set to `False`. This violates the semantic contract that a COO matrix with no duplicates should have `has_canonical_format=True`.

## Property-Based Test

```python
import scipy.sparse as sp
from hypothesis import given, strategies as st, settings

@settings(max_examples=500)
@given(
    n=st.integers(min_value=2, max_value=15),
    density=st.floats(min_value=0.1, max_value=0.5)
)
def test_coo_from_csr_canonical_property(n, density):
    coo = sp.random(n, n, density=density, format='coo', random_state=42)
    coo.sum_duplicates()

    csr = coo.tocsr()
    coo_from_csr = csr.tocoo()

    positions = list(zip(coo_from_csr.row, coo_from_csr.col))
    unique_positions = set(positions)
    has_duplicates = len(positions) != len(unique_positions)

    assert not has_duplicates, "COO from CSR should have no duplicates"
    assert coo_from_csr.has_canonical_format, \
        "COO from CSR should have has_canonical_format=True"
```

**Failing input**: Any CSR matrix converted to COO (e.g., `n=5, density=0.3`)

## Reproducing the Bug

```python
import scipy.sparse as sp

csr = sp.random(5, 5, density=0.3, format='csr', random_state=42)
coo = csr.tocoo()

positions = list(zip(coo.row, coo.col))
unique_positions = set(positions)

print(f"has_canonical_format: {coo.has_canonical_format}")
print(f"Total entries: {len(positions)}")
print(f"Unique positions: {len(unique_positions)}")
print(f"Has duplicates: {len(positions) != len(unique_positions)}")
```

**Output:**
```
has_canonical_format: False
Total entries: 8
Unique positions: 8
Has duplicates: False
```

The matrix has no duplicates (8 unique positions out of 8 total entries), yet `has_canonical_format` is `False`.

## Why This Is A Bug

According to scipy documentation and semantics, the `has_canonical_format` flag should indicate whether a COO matrix has:
1. No duplicate (i,j) entries
2. Entries sorted in row-major order (for some operations)

Since CSR format inherently has no duplicates (each (row, column) pair appears at most once), converting CSR→COO produces a COO matrix with no duplicates. The `has_canonical_format` flag should be set to `True` to reflect this fact.

This is a **contract violation** because:
- Users relying on `has_canonical_format` to check for duplicates will get false negatives
- Code that checks this flag before calling `sum_duplicates()` will unnecessarily call it on already-canonical matrices
- The flag's documented meaning (no duplicates) is not being maintained

## Fix

The bug is in the CSR-to-COO conversion code. When creating a COO matrix from CSR format, the `has_canonical_format` flag should be set to `True` since CSR format guarantees no duplicate entries.

The fix would be in the `tocoo()` method of CSR matrices (likely in `scipy/sparse/_compressed.py` or similar):

```diff
def tocoo(self, copy=False):
    ...
    coo = coo_matrix((data, (row, col)), shape=self.shape, dtype=self.dtype)
+   coo.has_canonical_format = True  # CSR has no duplicates
    return coo
```

This same fix should apply to CSC→COO conversion as well, since CSC format also has no duplicates.