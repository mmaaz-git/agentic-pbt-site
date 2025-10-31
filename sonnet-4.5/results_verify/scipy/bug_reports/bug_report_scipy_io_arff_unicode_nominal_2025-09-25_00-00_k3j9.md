# Bug Report: scipy.io.arff Unicode Nominal Values

**Target**: `scipy.io.arff.loadarff`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `loadarff` function crashes with `UnicodeEncodeError` when ARFF files contain nominal attribute values with non-ASCII Unicode characters. This occurs because the function uses ASCII byte strings ('S' dtype) instead of Unicode strings ('U' dtype) for nominal attributes.

## Property-Based Test

```python
from io import StringIO

import numpy as np
from hypothesis import assume, given, settings, strategies as st

from scipy.io import arff


def generate_arff_content(relation_name, attributes, data_rows):
    lines = [f"@relation {relation_name}", ""]

    for attr_name, attr_type in attributes:
        if isinstance(attr_type, list):
            attr_type_str = "{" + ",".join(attr_type) + "}"
        else:
            attr_type_str = attr_type
        lines.append(f"@attribute {attr_name} {attr_type_str}")

    lines.append("")
    lines.append("@data")

    for row in data_rows:
        lines.append(",".join(str(v) for v in row))

    return "\n".join(lines)


arff_relation_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() and not s[0].isdigit())

arff_attribute_name = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=15,
).filter(lambda s: s.strip() and not s[0].isdigit())

arff_nominal_value = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
).filter(lambda s: s.strip() and not s[0].isdigit())


@st.composite
def arff_attributes(draw):
    num_attrs = draw(st.integers(min_value=1, max_value=10))
    attrs = []
    seen_names = set()

    for _ in range(num_attrs):
        name = draw(arff_attribute_name)
        while name in seen_names:
            name = draw(arff_attribute_name)
        seen_names.add(name)

        attr_type_choice = draw(st.sampled_from(["numeric", "nominal"]))

        if attr_type_choice == "numeric":
            attrs.append((name, "numeric"))
        else:
            num_values = draw(st.integers(min_value=2, max_value=5))
            values = []
            seen_values = set()
            for _ in range(num_values):
                val = draw(arff_nominal_value)
                while val in seen_values:
                    val = draw(arff_nominal_value)
                seen_values.add(val)
                values.append(val)
            attrs.append((name, values))

    return attrs


@st.composite
def arff_data_row(draw, attributes):
    row = []
    for attr_name, attr_type in attributes:
        if attr_type == "numeric":
            value = draw(
                st.one_of(
                    st.floats(
                        allow_nan=False,
                        allow_infinity=False,
                        min_value=-1e10,
                        max_value=1e10,
                    ),
                    st.integers(min_value=-1000000, max_value=1000000),
                )
            )
            row.append(value)
        else:
            value = draw(st.sampled_from(attr_type))
            row.append(value)
    return row


@st.composite
def arff_file(draw):
    relation = draw(arff_relation_name)
    attributes = draw(arff_attributes())
    num_rows = draw(st.integers(min_value=0, max_value=20))
    rows = [draw(arff_data_row(attributes)) for _ in range(num_rows)]

    content = generate_arff_content(relation, attributes, rows)
    return content, relation, attributes, rows


@given(arff_file())
@settings(max_examples=200)
def test_metadata_names_types_length_consistency(arff_data):
    content, relation, attributes, rows = arff_data

    data, meta = arff.loadarff(StringIO(content))

    assert len(meta.names()) == len(meta.types())
```

**Failing input**: `('@relation A\n\n@attribute A {ª,A}\n\n@data\nª', 'A', [('A', ['ª', 'A'])], [['ª']])`

## Reproducing the Bug

```python
from io import StringIO

from scipy.io import arff

content = """@relation A

@attribute color {ª,blue}

@data
ª"""

f = StringIO(content)
data, meta = arff.loadarff(f)
```

**Output**:
```
UnicodeEncodeError: 'ascii' codec can't encode character '\xaa' in position 0: ordinal not in range(128)
```

## Why This Is A Bug

The ARFF format supports Unicode characters in nominal values, but `loadarff` crashes when attempting to load such files. This is because the function uses NumPy's 'S' (ASCII bytes) dtype for nominal attributes instead of 'U' (Unicode string) dtype. When NumPy tries to convert Unicode strings to ASCII bytes, it fails with a `UnicodeEncodeError`.

This affects real users who:
- Have data with international characters (e.g., names, places)
- Use scientific notation with special symbols (e.g., ª, µ, °)
- Work with multilingual datasets

## Fix

The fix is to change the dtype used for nominal attributes from 'S' (bytes) to 'U' (Unicode). This likely requires modifying the `NominalAttribute` class's `dtype` property in `_arffread.py`.

Based on the error location at line 871:
```python
data = np.array(a, [(a.name, a.dtype) for a in attr])
```

The attribute's `dtype` property needs to be changed. For nominal attributes, instead of returning something like `'S<n>'`, it should return `'U<n>'` where `<n>` is the maximum length of the nominal values.

**Example fix pattern** (exact implementation may vary based on the actual attribute class):
```diff
# In the NominalAttribute class (or equivalent)
 @property
 def dtype(self):
-    # Returns something like 'S10' for ASCII bytes
-    return f'S{self._max_len}'
+    # Use Unicode dtype instead
+    return f'U{self._max_len}'
```

This change ensures that nominal values with Unicode characters can be properly stored in the NumPy array without encoding errors.