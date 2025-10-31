# Bug Report: QueryDict.pop() Returns List While __getitem__ Returns Single Value

**Target**: `django.http.QueryDict.pop`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

QueryDict violates the dict protocol by having `pop()` and `__getitem__` return different types. `qd[key]` returns the last value as a string, but `qd.pop(key)` returns all values as a list. This inconsistency breaks the expected behavior where these operations should return the same type.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        DEFAULT_CHARSET='utf-8',
    )
    django.setup()

from hypothesis import given, strategies as st, settings
from django.http import QueryDict


@given(
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
    st.text(max_size=30)
)
@settings(max_examples=500)
def test_querydict_pop_getitem_consistency(key, value):
    qd = QueryDict(mutable=True)
    qd[key] = value

    retrieved_via_getitem = qd[key]
    type_via_getitem = type(retrieved_via_getitem)

    qd2 = QueryDict(mutable=True)
    qd2[key] = value
    popped = qd2.pop(key)
    type_via_pop = type(popped)

    assert type_via_getitem == type_via_pop, \
        f"qd[key] and qd.pop(key) should return the same type.\n" \
        f"qd[{key!r}] returned {type_via_getitem.__name__}: {retrieved_via_getitem!r}\n" \
        f"qd.pop({key!r}) returned {type_via_pop.__name__}: {popped!r}"
```

**Failing input**: `key='a', value=''` (or any key-value pair)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test', DEFAULT_CHARSET='utf-8')
    django.setup()

from django.http import QueryDict

qd = QueryDict(mutable=True)
qd['key'] = 'value'

print(f"qd['key'] = {qd['key']!r}")
print(f"Type: {type(qd['key']).__name__}")

qd2 = QueryDict(mutable=True)
qd2['key'] = 'value'
popped = qd2.pop('key')

print(f"\nqd.pop('key') = {popped!r}")
print(f"Type: {type(popped).__name__}")

print("\nThese should be the same type, but they're not!")
```

**Output:**
```
qd['key'] = 'value'
Type: str

qd.pop('key') = ['value']
Type: list

These should be the same type, but they're not!
```

## Why This Is A Bug

The Python dict protocol expects that if `d[key]` returns a value of type T, then `d.pop(key)` should also return type T. QueryDict violates this invariant:

- `QueryDict.__getitem__` explicitly returns the **last value** from the list (see docstring: "Return the last data value for this key")
- `QueryDict.pop` inherits from `MultiValueDict` which returns the **entire list**

This inconsistency can cause:
1. Type errors in code that expects `pop()` to return the same type as `__getitem__`
2. Unexpected behavior when migrating from regular dict to QueryDict
3. Bugs when refactoring code from `value = qd[key]; del qd[key]` to `value = qd.pop(key)`

## Fix

The fix should make `QueryDict.pop()` return the last value (a string) to match `__getitem__` behavior, while keeping a separate method (like `poplist()`) to get all values:

```diff
+++ b/django/http/request.py
@@ -615,7 +615,17 @@ class QueryDict(MultiValueDict):

     def pop(self, key, *args):
+        """
+        Remove specified key and return the corresponding last value.
+        If key is not found, return the default if given; otherwise,
+        raise a KeyError.
+        """
         self._assert_mutable()
-        return super().pop(key, *args)
+        try:
+            value_list = super().pop(key)
+            return value_list[-1] if value_list else []
+        except KeyError:
+            if args:
+                return args[0]
+            raise

     def popitem(self):
```

Alternatively, to maintain backward compatibility while fixing the inconsistency, Django could:
1. Add a deprecation warning to `pop()` about the behavior change
2. Add a new method `pop_list()` that explicitly returns all values
3. Eventually change `pop()` to return the last value

The key insight is that `QueryDict` already has this pattern with other methods:
- `__getitem__` returns last value
- `getlist()` returns all values
- Similarly, `pop()` should return last value and there should be a `poplist()` for all values