# Bug Report: django.http.QueryDict.pop Type Inconsistency with __getitem__

**Target**: `django.http.QueryDict.pop`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

QueryDict.pop() returns a list while __getitem__ returns a string for the same key, violating the dict protocol contract where these operations should return the same type.

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


if __name__ == "__main__":
    test_querydict_pop_getitem_consistency()
```

<details>

<summary>
**Failing input**: `key='a', value=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 43, in <module>
    test_querydict_pop_getitem_consistency()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 20, in test_querydict_pop_getitem_consistency
    st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=20),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/58/hypo.py", line 36, in test_querydict_pop_getitem_consistency
    assert type_via_getitem == type_via_pop, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: qd[key] and qd.pop(key) should return the same type.
qd['a'] returned str: ''
qd.pop('a') returned list: ['']
Falsifying example: test_querydict_pop_getitem_consistency(
    key='a',
    value='',
)
```
</details>

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

# Create a QueryDict with a key-value pair
qd = QueryDict(mutable=True)
qd['key'] = 'value'

# Access via __getitem__ (qd[key])
print(f"qd['key'] = {qd['key']!r}")
print(f"Type: {type(qd['key']).__name__}")

# Create another QueryDict for pop() test
qd2 = QueryDict(mutable=True)
qd2['key'] = 'value'

# Access via pop()
popped = qd2.pop('key')
print(f"\nqd.pop('key') = {popped!r}")
print(f"Type: {type(popped).__name__}")

# Demonstrate the inconsistency
print("\nThese should be the same type, but they're not!")
print(f"__getitem__ returns: {type(qd['key']).__name__}")
print(f"pop() returns: {type(popped).__name__}")

# Show what happens with multiple values
qd3 = QueryDict(mutable=True)
qd3.setlist('multi', ['first', 'second', 'third'])

print(f"\nFor multiple values:")
print(f"qd3['multi'] (via __getitem__) = {qd3['multi']!r}")
print(f"Type: {type(qd3['multi']).__name__}")

qd4 = QueryDict(mutable=True)
qd4.setlist('multi', ['first', 'second', 'third'])
multi_popped = qd4.pop('multi')
print(f"\nqd4.pop('multi') = {multi_popped!r}")
print(f"Type: {type(multi_popped).__name__}")

print("\nThe issue: __getitem__ returns the last value (string), but pop() returns the entire list!")
```

<details>

<summary>
Type inconsistency between __getitem__ and pop()
</summary>
```
qd['key'] = 'value'
Type: str

qd.pop('key') = ['value']
Type: list

These should be the same type, but they're not!
__getitem__ returns: str
pop() returns: list

For multiple values:
qd3['multi'] (via __getitem__) = 'third'
Type: str

qd4.pop('multi') = ['first', 'second', 'third']
Type: list

The issue: __getitem__ returns the last value (string), but pop() returns the entire list!
```
</details>

## Why This Is A Bug

This violates the fundamental dict protocol contract in Python where `d[key]` and `d.pop(key)` should return values of the same type. The inconsistency stems from QueryDict's inheritance hierarchy:

1. **QueryDict.__getitem__** (line 78-90 in django/utils/datastructures.py) explicitly returns the **last value** from the internal list:
   ```python
   def __getitem__(self, key):
       """Return the last data value for this key, or [] if it's an empty list"""
       list_ = super().__getitem__(key)
       return list_[-1]  # Returns the last value
   ```

2. **QueryDict.pop** (line 651-653 in django/http/request.py) simply delegates to MultiValueDict's pop, which inherits from dict.pop:
   ```python
   def pop(self, key, *args):
       self._assert_mutable()
       return super().pop(key, *args)  # Returns the entire list
   ```

3. **MultiValueDict** stores all values as lists internally but presents a dual interface:
   - Single-value methods (`__getitem__`, `get`) return the last value
   - List methods (`getlist`, `lists`) return all values
   - However, `pop()` breaks this pattern by returning all values

This API asymmetry causes real-world problems:
- **Type errors**: Code expecting a string from pop() receives a list unexpectedly
- **Refactoring hazards**: `value = qd[key]; del qd[key]` behaves differently from `value = qd.pop(key)`
- **Migration issues**: Code moving from standard dict to QueryDict breaks silently
- **Violates Principle of Least Surprise**: Developers familiar with Python's dict expect consistent behavior

## Relevant Context

Django's MultiValueDict establishes a clear API pattern for handling single vs. multiple values:

| Operation | Single Value Method | Multiple Values Method |
|-----------|-------------------|----------------------|
| Get       | `__getitem__(key)` → str | `getlist(key)` → list |
| Set       | `__setitem__(key, val)` | `setlist(key, list)` |
| Pop       | **Missing!** | `pop(key)` → list |

The pop() method should follow this established pattern. Django documentation for QueryDict states that `__getitem__` returns "the last data value for this key", but the pop() behavior is undocumented, leaving developers to discover this inconsistency through runtime errors.

Related Django source files:
- [django/utils/datastructures.py](https://github.com/django/django/blob/main/django/utils/datastructures.py) - MultiValueDict implementation
- [django/http/request.py](https://github.com/django/django/blob/main/django/http/request.py) - QueryDict implementation

## Proposed Fix

```diff
--- a/django/http/request.py
+++ b/django/http/request.py
@@ -649,9 +649,30 @@ class QueryDict(MultiValueDict):
         super().appendlist(key, value)

     def pop(self, key, *args):
+        """
+        Remove specified key and return the corresponding last value.
+        If key is not found, return the default if given; otherwise,
+        raise a KeyError.
+        """
         self._assert_mutable()
-        return super().pop(key, *args)
+        try:
+            value_list = dict.pop(self, key)
+            return value_list[-1] if value_list else []
+        except KeyError:
+            if args:
+                return args[0]
+            raise
+
+    def poplist(self, key, *args):
+        """
+        Remove specified key and return the corresponding list of values.
+        If key is not found, return the default if given; otherwise,
+        raise a KeyError.
+        """
+        self._assert_mutable()
+        return dict.pop(self, key, *args)

     def popitem(self):
         self._assert_mutable()
         return super().popitem()
```