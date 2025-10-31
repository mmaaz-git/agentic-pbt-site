# Bug Report: Flask Config.from_mapping Misleading Docstring

**Target**: `flask.Config.from_mapping`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Config.from_mapping()` method has a misleading docstring that claims it "Updates the config like :meth:`update` ignoring items with non-upper keys." However, the `update()` method does NOT ignore non-uppercase keys, making this comparison incorrect and potentially confusing to users.

## Property-Based Test

```python
from flask import Flask
from hypothesis import given, strategies as st

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.integers(),
        min_size=1
    )
)
def test_config_update_only_uppercase(mapping):
    app = Flask(__name__)
    config = app.config

    initial_keys = set(config.keys())
    config.update(mapping)

    for key in mapping:
        if key.isupper():
            assert key in config
            assert config[key] == mapping[key]
        else:
            if key not in initial_keys:
                assert key not in config
```

**Failing input**: `{'0': 0}` (or any mapping with non-uppercase keys)

## Reproducing the Bug

```python
from flask import Flask

app = Flask(__name__)
config = app.config

config.update({'lowercase_key': 'value1', 'UPPERCASE_KEY': 'value2'})

print("'lowercase_key' in config:", 'lowercase_key' in config)
print("'UPPERCASE_KEY' in config:", 'UPPERCASE_KEY' in config)

app2 = Flask(__name__)
config2 = app2.config

config2.from_mapping({'lowercase_key': 'value1', 'UPPERCASE_KEY': 'value2'})

print("'lowercase_key' in config2:", 'lowercase_key' in config2)
print("'UPPERCASE_KEY' in config2:", 'UPPERCASE_KEY' in config2)
```

**Output:**
```
'lowercase_key' in config: True
'UPPERCASE_KEY' in config: True
'lowercase_key' in config2: False
'UPPERCASE_KEY' in config2: True
```

## Why This Is A Bug

The `from_mapping()` docstring states:
> "Updates the config like :meth:`update` ignoring items with non-upper keys."

This implies that `update()` normally accepts all keys, and `from_mapping()` is like `update()` but with filtering. However, the phrasing "like :meth:`update`" suggests they behave similarly, which is misleading since `update()` does not filter any keys.

Users reading this docstring might incorrectly assume that `update()` also filters non-uppercase keys, or they might be confused about the difference between the two methods.

## Fix

Update the docstring to be more accurate:

```diff
--- a/flask/config.py
+++ b/flask/config.py
@@ -155,8 +155,9 @@ class Config(dict):
     def from_mapping(
         self, mapping: t.Mapping[str, t.Any] | None = None, **kwargs: t.Any
     ) -> bool:
-        """Updates the config like :meth:`update` ignoring items with
-        non-upper keys.
+        """Updates the config from a mapping, only adding items with
+        uppercase keys. Unlike :meth:`update`, which accepts all keys,
+        this method ignores items with non-uppercase keys.

         :return: Always returns ``True``.
```