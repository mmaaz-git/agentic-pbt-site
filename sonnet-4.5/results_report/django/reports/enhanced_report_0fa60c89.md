# Bug Report: django.db.migrations.serializer.DictionarySerializer Crashes on Mixed-Type Dictionary Keys

**Target**: `django.db.migrations.serializer.DictionarySerializer`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `DictionarySerializer.serialize()` method in Django's migration system crashes with a `TypeError` when attempting to serialize dictionaries containing keys of different types (e.g., integers and strings), because it sorts dictionary items without providing a key function that can handle incompatible types.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from django.db.migrations.serializer import DictionarySerializer

@given(st.dictionaries(
    st.one_of(st.integers(), st.text()),
    st.one_of(st.integers(), st.text()),
    max_size=10
))
def test_dictionary_serializer_deterministic(d):
    serializer = DictionarySerializer(d)
    result, _ = serializer.serialize()
```

<details>

<summary>
**Failing input**: `{0: 0, '': 0}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 15, in <module>
    test_dictionary_serializer_deterministic()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 5, in test_dictionary_serializer_deterministic
    st.one_of(st.integers(), st.text()),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/12/hypo.py", line 11, in test_dictionary_serializer_deterministic
    result, _ = serializer.serialize()
                ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/migrations/serializer.py", line 129, in serialize
    for k, v in sorted(self.value.items()):
                ~~~~~~^^^^^^^^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'str' and 'int'
Falsifying example: test_dictionary_serializer_deterministic(
    d={0: 0, '': 0},
)
```
</details>

## Reproducing the Bug

```python
from django.db.migrations.serializer import DictionarySerializer

# Test case with mixed-type dictionary keys
d = {0: 0, '': 0}
serializer = DictionarySerializer(d)

try:
    result, imports = serializer.serialize()
    print(f"Success: {result}")
    print(f"Imports: {imports}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TypeError: '<' not supported between instances of 'str' and 'int'
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/12/repo.py", line 8, in <module>
    result, imports = serializer.serialize()
                      ~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/django/db/migrations/serializer.py", line 129, in serialize
    for k, v in sorted(self.value.items()):
                ~~~~~~^^^^^^^^^^^^^^^^^^^^
TypeError: '<' not supported between instances of 'str' and 'int'
TypeError: '<' not supported between instances of 'str' and 'int'
```
</details>

## Why This Is A Bug

This is a legitimate bug in Django's migration serialization system. The `DictionarySerializer` class is registered to handle all `dict` type objects in the migration system (line 348 of serializer.py), yet it crashes on valid Python dictionaries that contain mixed-type keys.

The root cause is in line 129 of `/django/db/migrations/serializer.py`, where the code attempts to sort dictionary items for deterministic output:
```python
for k, v in sorted(self.value.items()):
```

In Python 3, the `sorted()` function cannot directly compare values of incompatible types (e.g., `int` and `str`). When Python tries to compare `0 < ''`, it raises a `TypeError` because these types are not comparable. This is a fundamental change from Python 2, where such comparisons were allowed (though arbitrary).

Mixed-type dictionaries are perfectly valid in Python and can legitimately appear in Django applications, such as:
- Configuration dictionaries mixing numeric codes with string identifiers
- Field choices that combine database IDs (integers) with string codes
- Mapping tables that use both numeric and string keys for different purposes
- JSON data imported from external sources

The serializer should handle all valid Python dictionaries gracefully, not crash on a subset of them.

## Relevant Context

The `DictionarySerializer` class has no documentation or docstring explaining its limitations. The sorting is performed to ensure deterministic output in migration files, which is important for version control and reproducibility. However, the current implementation assumes all dictionary keys are mutually comparable, which is not true in Python 3.

The Django documentation at https://docs.djangoproject.com/en/stable/topics/migrations/#migration-serializing mentions migration serialization but does not specify any restrictions on dictionary key types. Users would reasonably expect that any valid Python dictionary can be serialized for migrations.

This bug affects any Django application that:
- Uses dictionaries with mixed-type keys in model field defaults
- Passes mixed-type dictionaries as field options or parameters
- Has custom migration operations that include mixed-type dictionaries

Code location: `/django/db/migrations/serializer.py:129`

## Proposed Fix

```diff
--- a/django/db/migrations/serializer.py
+++ b/django/db/migrations/serializer.py
@@ -126,7 +126,7 @@ class DictionarySerializer(BaseSerializer):
     def serialize(self):
         imports = set()
         strings = []
-        for k, v in sorted(self.value.items()):
+        for k, v in sorted(self.value.items(), key=lambda x: (type(x[0]).__name__, x[0])):
             k_string, k_imports = serializer_factory(k).serialize()
             v_string, v_imports = serializer_factory(v).serialize()
             imports.update(k_imports)
```

This fix sorts dictionary items first by the type name (grouping same-type keys together), then by the actual value within each type group. This ensures:
1. Deterministic ordering is maintained (the original goal)
2. No TypeError occurs when comparing incompatible types
3. All valid Python dictionaries can be serialized
4. Backward compatibility is preserved (homogeneous dictionaries sort the same way)