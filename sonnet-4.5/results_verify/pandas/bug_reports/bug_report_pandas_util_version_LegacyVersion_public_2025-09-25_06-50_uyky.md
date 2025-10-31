# Bug Report: pandas.util.version LegacyVersion.public Property

**Target**: `pandas.util.version.LegacyVersion.public`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `LegacyVersion.public` property does not remove the local version identifier (part after '+'), while `Version.public` does, causing inconsistent behavior across the version API.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas.util.version as version_module

@st.composite
def valid_version_strings(draw):
    ep = draw(st.one_of(st.just(None), st.integers(min_value=0, max_value=10)))
    rel = draw(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=5))
    has_pre = draw(st.booleans())
    has_post = draw(st.booleans())
    has_dev = draw(st.booleans())
    has_local = draw(st.booleans())

    parts = []
    if ep and ep != 0:
        parts.append(f"{ep}!")
    parts.append(".".join(str(x) for x in rel))

    if has_pre:
        pre_type = draw(st.sampled_from(["a", "b", "rc"]))
        pre_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f"{pre_type}{pre_num}")

    if has_post:
        post_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f".post{post_num}")

    if has_dev:
        dev_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f".dev{dev_num}")

    if has_local:
        local = draw(st.lists(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd')), min_size=1, max_size=10), min_size=1, max_size=3))
        parts.append("+" + ".".join(local))

    return "".join(parts)

@given(valid_version_strings())
@settings(max_examples=500)
def test_public_version_no_local(version_str):
    v = version_module.parse(version_str)
    public = v.public
    assert "+" not in public, f"public version contains local part: {public}"
```

**Failing input**: `'0.dev0+µ'`

## Reproducing the Bug

```python
import pandas.util.version

v1 = pandas.util.version.parse("1.0.0+local")
print(f"Version.public: '{v1.public}'")
print(f"Has '+': {'+' in v1.public}")

v2 = pandas.util.version.parse("not-pep440+local")
print(f"LegacyVersion.public: '{v2.public}'")
print(f"Has '+': {'+' in v2.public}")
```

Output:
```
Version.public: '1.0.0'
Has '+': False
LegacyVersion.public: 'not-pep440+local'
Has '+': True
```

## Why This Is A Bug

The `public` property is documented to return the public version identifier, which excludes local version identifiers. The `Version` class correctly implements this by splitting on '+' and returning only the first part. However, `LegacyVersion.public` just returns the raw version string unchanged, including any local identifiers.

This creates inconsistent behavior: the same property on objects that share a common base class (`_BaseVersion`) and API behaves differently. Users cannot rely on `.public` to consistently strip local identifiers.

## Fix

```diff
--- a/pandas/util/version/__init__.py
+++ b/pandas/util/version/__init__.py
@@ -199,7 +199,7 @@ class LegacyVersion(_BaseVersion):

     @property
     def public(self) -> str:
-        return self._version
+        return self._version.split("+", 1)[0]

     @property
     def base_version(self) -> str:
```