# Bug Report: dask.dataframe Series.str.upper() Incorrect Unicode Handling for German ß Character

**Target**: `dask.dataframe.Series.str.upper()`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`dask.dataframe.Series.str.upper()` produces different results than pandas for the German ß character, converting it to capital ẞ (U+1E9E) instead of the traditional 'SS' that pandas returns.

## Property-Based Test

```python
import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.text(min_size=1, max_size=10),
        min_size=5,
        max_size=30
    )
)
@settings(max_examples=50)
def test_str_upper_matches_pandas(strings):
    pdf = pd.DataFrame({'text': strings})
    ddf = dd.from_pandas(pdf, npartitions=2)

    pandas_result = pdf['text'].str.upper()
    dask_result = ddf['text'].str.upper().compute()

    for i in range(len(strings)):
        assert pandas_result.iloc[i] == dask_result.iloc[i], \
            f"str.upper() mismatch for '{strings[i]}': pandas='{pandas_result.iloc[i]}', dask='{dask_result.iloc[i]}'"
```

<details>

<summary>
**Failing input**: `['0', '0', '0', '0', 'ß']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 28, in <module>
    test_str_upper_matches_pandas()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 7, in test_str_upper_matches_pandas
    st.lists(

  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/29/hypo.py", line 22, in test_str_upper_matches_pandas
    assert pandas_result.iloc[i] == dask_result.iloc[i], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: str.upper() mismatch for 'ß': pandas='SS', dask='ẞ'
Falsifying example: test_str_upper_matches_pandas(
    strings=['0', '0', '0', '0', 'ß'],
)
Explanation:
    These lines were always and only run by failing examples:
        /home/npc/pbt/agentic-pbt/worker_/29/hypo.py:23
```
</details>

## Reproducing the Bug

```python
import pandas as pd
import dask.dataframe as dd

# Create test data with German ß character
pdf = pd.DataFrame({'text': ['ß']})
ddf = dd.from_pandas(pdf, npartitions=1)

# Apply str.upper() to both
pandas_result = pdf['text'].str.upper().iloc[0]
dask_result = ddf['text'].str.upper().compute().iloc[0]

# Display results
print(f"Input: 'ß'")
print(f"Pandas: '{pandas_result}'")
print(f"Dask:   '{dask_result}'")

# Show the actual difference
print(f"\nPandas result equals 'SS': {pandas_result == 'SS'}")
print(f"Dask result equals 'ẞ': {dask_result == 'ẞ'}")
print(f"Dask result equals 'SS': {dask_result == 'SS'}")

# Additional test with German words
pdf_words = pd.DataFrame({'text': ['Straße', 'groß', 'weiß']})
ddf_words = dd.from_pandas(pdf_words, npartitions=1)

print("\nAdditional German words:")
for i in range(len(pdf_words)):
    pandas_upper = pdf_words['text'].str.upper().iloc[i]
    dask_upper = ddf_words['text'].str.upper().compute().iloc[i]
    print(f"Input: '{pdf_words['text'].iloc[i]}' -> Pandas: '{pandas_upper}', Dask: '{dask_upper}'")
```

<details>

<summary>
Output demonstrating the discrepancy
</summary>
```
Input: 'ß'
Pandas: 'SS'
Dask:   'ẞ'

Pandas result equals 'SS': True
Dask result equals 'ẞ': True
Dask result equals 'SS': False

Additional German words:
Input: 'Straße' -> Pandas: 'STRASSE', Dask: 'STRAẞE'
Input: 'groß' -> Pandas: 'GROSS', Dask: 'GROẞ'
Input: 'weiß' -> Pandas: 'WEISS', Dask: 'WEIẞ'
```
</details>

## Why This Is A Bug

This violates the expected behavior and API contract between dask and pandas. The issue stems from the following:

1. **API Contract Violation**: Dask explicitly states in its documentation that "Dask DataFrame copies pandas" and "The API is the same. The execution is the same." The `str.upper()` docstring is copied directly from pandas and states it is "Equivalent to :meth:`str.upper`" (Python's standard string method).

2. **Undocumented Behavior**: While the docstring mentions "Some inconsistencies with the Dask version may exist," it does not specify that Unicode case conversion differs from pandas. This is a significant behavioral difference that affects real-world text processing.

3. **Different String Backends**: The root cause is that dask uses PyArrow strings by default (dtype='string'), while pandas typically uses object dtype. PyArrow's `utf8_upper()` converts 'ß' to 'ẞ' (capital eszett, U+1E9E), while Python's `str.upper()` converts 'ß' to 'SS' following traditional German orthographic conventions.

4. **Impact on German Text Processing**: This affects any application processing German text that relies on the traditional 'ß' → 'SS' conversion, which has been the standard in German typography for centuries and is still widely expected, despite the capital ẞ being officially recognized since 2017.

## Relevant Context

- **Dask version**: 2025.9.1
- **PyArrow version**: 20.0.0
- **Python version**: 3.13.2
- **Default string dtype in Dask**: 'string' (PyArrow-backed)
- **Default string dtype in Pandas**: 'object' (Python str)

The Unicode capital eszett (ẞ, U+1E9E) was added in Unicode 5.1 (2008) and officially recognized in German orthography in 2017. While technically valid, the traditional 'ß' → 'SS' mapping remains more common and is what Python's `str.upper()` implements for backward compatibility.

Documentation links:
- Dask DataFrame documentation: https://docs.dask.org/en/stable/dataframe.html
- Dask str accessor source: `/home/npc/miniconda/lib/python3.13/site-packages/dask/dataframe/dask_expr/_str_accessor.py:72`

## Proposed Fix

The fix would involve ensuring dask's string operations match pandas behavior when using the default configuration. Here are potential approaches:

1. **Configure PyArrow to use Python-compatible case mapping** (if possible)
2. **Override the upper() method for German characters specifically**
3. **Document the difference and provide configuration option**

Since the issue is in the PyArrow backend, a workaround for users is to configure dask to use object dtype instead:

```diff
import dask
import dask.dataframe as dd

+ # Workaround: Force dask to use object dtype like pandas default
+ dask.config.set({"dataframe.convert-string": False})

pdf = pd.DataFrame({'text': ['ß']})
ddf = dd.from_pandas(pdf, npartitions=1)
# Now dask will match pandas behavior
```

A proper fix would require modifying dask's string accessor to handle this case specially or configuring the PyArrow string operations to match Python's Unicode case mapping rules.