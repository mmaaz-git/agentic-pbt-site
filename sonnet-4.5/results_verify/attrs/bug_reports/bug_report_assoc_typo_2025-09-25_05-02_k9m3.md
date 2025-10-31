# Bug Report: assoc Function Docstring Typo

**Target**: `attr.assoc`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `assoc()` function docstring contains a typo: "du to" instead of "due to".

## Property-Based Test

N/A - This is a documentation typo, not a functional bug.

**Failing input**: N/A

## Reproducing the Bug

```python
import attr

print(attr.assoc.__doc__)
```

The docstring contains:
```
Use `attrs.evolve` instead if you can. This function will not be
removed du to the slightly different approach compared to
```

## Why This Is A Bug

The word "du" at line 383 of `/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages/attr/_funcs.py` should be "due". This is a simple typo in the documentation that makes the sentence grammatically incorrect.

## Fix

```diff
--- a/attr/_funcs.py
+++ b/attr/_funcs.py
@@ -380,7 +380,7 @@ def assoc(inst, **changes):

     ..  deprecated:: 17.1.0
         Use `attrs.evolve` instead if you can. This function will not be
-        removed du to the slightly different approach compared to
+        removed due to the slightly different approach compared to
         `attrs.evolve`, though.
     """
     new = copy.copy(inst)
```