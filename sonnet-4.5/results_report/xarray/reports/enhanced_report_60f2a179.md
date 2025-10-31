# Bug Report: xarray.core.dtypes AlwaysGreaterThan/AlwaysLessThan Total Ordering Violation

**Target**: `xarray.core.dtypes.AlwaysGreaterThan` and `xarray.core.dtypes.AlwaysLessThan`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AlwaysGreaterThan` and `AlwaysLessThan` classes violate the antisymmetry property of total ordering. When two instances are equal according to `__eq__`, they incorrectly report being greater than (or less than) each other, violating the mathematical contract required by `@functools.total_ordering`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

@given(st.just(None))  # We don't need any parameters, just run once
def test_always_greater_than_total_ordering(dummy):
    """Test that AlwaysGreaterThan satisfies total ordering properties."""
    inf1 = AlwaysGreaterThan()
    inf2 = AlwaysGreaterThan()

    # Test reflexivity: a == a
    assert inf1 == inf1

    # Test symmetry: if a == b then b == a
    assert inf1 == inf2
    assert inf2 == inf1

    # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
    assert inf1 == inf2
    assert not (inf1 != inf2)
    assert not (inf1 < inf2), "AlwaysGreaterThan instances should not be less than each other"
    assert not (inf1 > inf2), f"AlwaysGreaterThan instances should not be greater than each other when equal. Got inf1 > inf2 = {inf1 > inf2}"

    # Test that <= and >= work correctly for equal values
    assert inf1 <= inf2, "inf1 <= inf2 should be True when they are equal"
    assert inf1 >= inf2, "inf1 >= inf2 should be True when they are equal"

@given(st.just(None))  # We don't need any parameters, just run once
def test_always_less_than_total_ordering(dummy):
    """Test that AlwaysLessThan satisfies total ordering properties."""
    ninf1 = AlwaysLessThan()
    ninf2 = AlwaysLessThan()

    # Test reflexivity: a == a
    assert ninf1 == ninf1

    # Test symmetry: if a == b then b == a
    assert ninf1 == ninf2
    assert ninf2 == ninf1

    # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
    assert ninf1 == ninf2
    assert not (ninf1 != ninf2)
    assert not (ninf1 < ninf2), f"AlwaysLessThan instances should not be less than each other when equal. Got ninf1 < ninf2 = {ninf1 < ninf2}"
    assert not (ninf1 > ninf2), "AlwaysLessThan instances should not be greater than each other"

    # Test that <= and >= work correctly for equal values
    assert ninf1 <= ninf2, "ninf1 <= ninf2 should be True when they are equal"
    assert ninf1 >= ninf2, "ninf1 >= ninf2 should be True when they are equal"

if __name__ == "__main__":
    # Run the tests
    test_always_greater_than_total_ordering()
    test_always_less_than_total_ordering()
```

<details>

<summary>
**Failing input**: `None` (any two instances of the same class fail)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/21
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 2 items

hypo.py::test_always_greater_than_total_ordering FAILED                  [ 50%]
hypo.py::test_always_less_than_total_ordering FAILED                     [100%]

=================================== FAILURES ===================================
___________________ test_always_greater_than_total_ordering ____________________

    @given(st.just(None))  # We don't need any parameters, just run once
>   def test_always_greater_than_total_ordering(dummy):
                   ^^^

hypo.py:5:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

dummy = None

    @given(st.just(None))  # We don't need any parameters, just run once
    def test_always_greater_than_total_ordering(dummy):
        """Test that AlwaysGreaterThan satisfies total ordering properties."""
        inf1 = AlwaysGreaterThan()
        inf2 = AlwaysGreaterThan()

        # Test reflexivity: a == a
        assert inf1 == inf1

        # Test symmetry: if a == b then b == a
        assert inf1 == inf2
        assert inf2 == inf1

        # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
        assert inf1 == inf2
        assert not (inf1 != inf2)
        assert not (inf1 < inf2), "AlwaysGreaterThan instances should not be less than each other"
>       assert not (inf1 > inf2), f"AlwaysGreaterThan instances should not be greater than each other when equal. Got inf1 > inf2 = {inf1 > inf2}"
E       AssertionError: AlwaysGreaterThan instances should not be greater than each other when equal. Got inf1 > inf2 = True
E       assert not <xarray.core.dtypes.AlwaysGreaterThan object at 0x71817e5aa060> > <xarray.core.dtypes.AlwaysGreaterThan object at 0x71817e5aa190>
E       Falsifying example: test_always_greater_than_total_ordering(
E           dummy=None,
E       )

hypo.py:21: AssertionError
_____________________ test_always_less_than_total_ordering _____________________

    @given(st.just(None))  # We don't need any parameters, just run once
>   def test_always_less_than_total_ordering(dummy):
                   ^^^

hypo.py:28:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

dummy = None

    @given(st.just(None))  # We don't need any parameters, just run once
    def test_always_less_than_total_ordering(dummy):
        """Test that AlwaysLessThan satisfies total ordering properties."""
        ninf1 = AlwaysLessThan()
        ninf2 = AlwaysLessThan()

        # Test reflexivity: a == a
        assert ninf1 == ninf1

        # Test symmetry: if a == b then b == a
        assert ninf1 == ninf2
        assert ninf2 == ninf1

        # Test antisymmetry: if a == b, then not (a > b) and not (a < b)
        assert ninf1 == ninf2
        assert not (ninf1 != ninf2)
>       assert not (ninf1 < ninf2), f"AlwaysLessThan instances should not be less than each other when equal. Got ninf1 < ninf2 = {ninf1 < ninf2}"
E       AssertionError: AlwaysLessThan instances should not be less than each other when equal. Got ninf1 < ninf2 = True
E       assert not <xarray.core.dtypes.AlwaysLessThan object at 0x71817e5ab6f0> < <xarray.core.dtypes.AlwaysLessThan object at 0x71817e5ab490>
E       Falsifying example: test_always_less_than_total_ordering(
E           dummy=None,
E       )

hypo.py:43: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_always_greater_than_total_ordering - AssertionError: Alw...
FAILED hypo.py::test_always_less_than_total_ordering - AssertionError: Always...
============================== 2 failed in 0.45s ===============================
```
</details>

## Reproducing the Bug

```python
from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan

print("Testing AlwaysGreaterThan:")
print("=" * 40)
inf1 = AlwaysGreaterThan()
inf2 = AlwaysGreaterThan()

print(f"inf1 == inf2: {inf1 == inf2}")  # Should be True
print(f"inf1 != inf2: {inf1 != inf2}")  # Should be False
print(f"inf1 > inf2: {inf1 > inf2}")    # Should be False when equal (BUG: returns True)
print(f"inf1 < inf2: {inf1 < inf2}")    # Should be False when equal
print(f"inf1 >= inf2: {inf1 >= inf2}")  # Should be True when equal
print(f"inf1 <= inf2: {inf1 <= inf2}")  # Should be True when equal (BUG: returns False)

print("\nTesting AlwaysLessThan:")
print("=" * 40)
ninf1 = AlwaysLessThan()
ninf2 = AlwaysLessThan()

print(f"ninf1 == ninf2: {ninf1 == ninf2}")  # Should be True
print(f"ninf1 != ninf2: {ninf1 != ninf2}")  # Should be False
print(f"ninf1 < ninf2: {ninf1 < ninf2}")    # Should be False when equal (BUG: returns True)
print(f"ninf1 > ninf2: {ninf1 > ninf2}")    # Should be False when equal
print(f"ninf1 <= ninf2: {ninf1 <= ninf2}")  # Should be True when equal
print(f"ninf1 >= ninf2: {ninf1 >= ninf2}")  # Should be True when equal (BUG: returns False)

print("\nTesting comparisons with other values:")
print("=" * 40)
print(f"inf1 > 100: {inf1 > 100}")         # Should be True
print(f"inf1 > 'string': {inf1 > 'string'}")  # Should be True
print(f"ninf1 < 100: {ninf1 < 100}")       # Should be True
print(f"ninf1 < 'string': {ninf1 < 'string'}")  # Should be True
```

<details>

<summary>
Demonstration of the bug with actual output showing violations
</summary>
```
Testing AlwaysGreaterThan:
========================================
inf1 == inf2: True
inf1 != inf2: False
inf1 > inf2: True
inf1 < inf2: False
inf1 >= inf2: True
inf1 <= inf2: False

Testing AlwaysLessThan:
========================================
ninf1 == ninf2: True
ninf1 != ninf2: False
ninf1 < ninf2: True
ninf1 > ninf2: False
ninf1 <= ninf2: True
ninf1 >= ninf2: False

Testing comparisons with other values:
========================================
inf1 > 100: True
inf1 > 'string': True
ninf1 < 100: True
ninf1 < 'string': True
```
</details>

## Why This Is A Bug

The `@functools.total_ordering` decorator requires that decorated classes satisfy the mathematical properties of a total order, specifically:

1. **Antisymmetry**: If `a ≤ b` and `b ≤ a`, then `a = b`. Equivalently, if `a == b`, then `not (a < b)` and `not (a > b)`.

2. **Transitivity**: If `a ≤ b` and `b ≤ c`, then `a ≤ c`.

3. **Totality**: Either `a ≤ b` or `b ≤ a` (or both).

The current implementation violates antisymmetry:
- When `inf1 == inf2` (both are `AlwaysGreaterThan` instances), `inf1 > inf2` returns `True` instead of `False`
- When `ninf1 == ninf2` (both are `AlwaysLessThan` instances), `ninf1 < ninf2` returns `True` instead of `False`

This happens because:
- `AlwaysGreaterThan.__gt__()` unconditionally returns `True`, even when comparing with another `AlwaysGreaterThan` instance
- `AlwaysLessThan.__lt__()` unconditionally returns `True`, even when comparing with another `AlwaysLessThan` instance

## Relevant Context

These classes are used internally in xarray as singleton sentinels to represent positive and negative infinity for object-type arrays (where `np.inf` and `-np.inf` cannot be used directly):

- Located in `/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages/xarray/core/dtypes.py:18-37`
- Exported as constants: `INF = AlwaysGreaterThan()` and `NINF = AlwaysLessThan()`
- Used in `get_pos_infinity()` and `get_neg_infinity()` functions for object dtypes
- Comment in source: "# Equivalence to np.inf (-np.inf) for object-type"

The violation of total ordering can cause:
- Incorrect sorting results when these sentinels appear in collections
- Inconsistent behavior in algorithms that rely on comparison transitivity
- Violations of expected mathematical properties for infinity representations

Python documentation for `functools.total_ordering`: https://docs.python.org/3/library/functools.html#functools.total_ordering

## Proposed Fix

```diff
@functools.total_ordering
class AlwaysGreaterThan:
    def __gt__(self, other):
+       if isinstance(other, type(self)):
+           return False
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))


@functools.total_ordering
class AlwaysLessThan:
    def __lt__(self, other):
+       if isinstance(other, type(self)):
+           return False
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))
```