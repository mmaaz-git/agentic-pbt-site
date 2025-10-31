# Bug Report: pydantic.experimental.pipeline String Transform Chaining Fails

**Target**: `pydantic.experimental.pipeline._apply_transform`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When chaining string transformations using `transform(str.lower).str_upper()`, only the first transformation is applied, violating the documented behavior that each transform should "Transform the output of the previous step."

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

@given(st.text())
@settings(max_examples=200)
def test_str_lower_then_upper(text):
    class Model(BaseModel):
        value: Annotated[str, transform(str.lower).str_upper()]

    m = Model(value=text)
    assert m.value == text.upper(), f"Expected {text.upper()!r} but got {m.value!r}"

test_str_lower_then_upper()
```

<details>

<summary>
**Failing input**: `text='A'`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 15, in <module>
    test_str_lower_then_upper()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 7, in test_str_lower_then_upper
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 13, in test_str_lower_then_upper
    assert m.value == text.upper(), f"Expected {text.upper()!r} but got {m.value!r}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'A' but got 'a'
Falsifying example: test_str_lower_then_upper(
    text='A',
)
```
</details>

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, transform(str.lower).str_upper()]

m = Model(value="ABC")
print(f"Input: 'ABC'")
print(f"Expected output: 'ABC' (first apply lower() to get 'abc', then apply upper() to get 'ABC')")
print(f"Actual output: '{m.value}'")
```

<details>

<summary>
Bug demonstration shows only lower() is applied
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Input: 'ABC'
Expected output: 'ABC' (first apply lower() to get 'abc', then apply upper() to get 'ABC')
Actual output: 'abc'
```
</details>

## Why This Is A Bug

The pipeline API documentation explicitly states that `transform` should "Transform the output of the previous step." When chaining transformations like `transform(str.lower).str_upper()`, the expected behavior is sequential application:

1. First, apply `str.lower` to the input "ABC" → "abc"
2. Then, apply `str.upper` to the result of step 1 → "ABC"

However, the actual implementation in `_apply_transform` (lines 419-441 in pipeline.py) optimizes string operations by setting schema flags directly. When multiple string transformations are chained:

1. The first `transform(str.lower)` sets `to_lower: true` on the schema
2. The second `.str_upper()` calls `transform(str.upper)`, which sets `to_upper: true` on the same schema
3. The resulting schema has both flags set simultaneously: `{'to_lower': true, 'to_upper': true}`
4. Pydantic-core appears to only apply the `to_lower` flag, ignoring `to_upper`

This violates the fundamental promise of the pipeline API that transformations compose sequentially.

## Relevant Context

The `_apply_transform` function at lines 427-439 contains the problematic optimization:

```python
if s['type'] == 'str':
    if func is str.strip:
        s = s.copy()
        s['strip_whitespace'] = True
        return s
    elif func is str.lower:
        s = s.copy()
        s['to_lower'] = True
        return s
    elif func is str.upper:
        s = s.copy()
        s['to_upper'] = True
        return s
```

This optimization assumes only one string transformation will be applied, but doesn't account for chaining multiple string transformations. The schema ends up with conflicting flags that pydantic-core cannot properly interpret.

Documentation reference: The `transform` method docstring (line 138) states "Transform the output of the previous step," establishing the contract for sequential transformation.

## Proposed Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -424,7 +424,9 @@ def _apply_transform(
     if s is None:
         return cs.no_info_plain_validator_function(func)

-    if s['type'] == 'str':
+    # Only use schema-level optimizations if no conflicting transformation is already applied
+    # This ensures chained string transformations work correctly
+    if s['type'] == 'str' and not (s.get('to_lower') or s.get('to_upper')):
         if func is str.strip:
             s = s.copy()
             s['strip_whitespace'] = True
```