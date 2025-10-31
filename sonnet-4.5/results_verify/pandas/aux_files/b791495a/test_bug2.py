#!/usr/bin/env python3
"""Test the purported bug in pandas categorical null value handling in interchange protocol."""

import pandas as pd
import numpy as np
from pandas.api.interchange import from_dataframe

# Multiple test cases to verify the bug
test_cases = [
    ["cat1", None],
    ["cat1", "cat2", None],
    [None, "cat1"],
    ["cat1", None, "cat2", None],
    ["cat1", "cat2", "cat3", None, None, "cat1"],
]

print("Testing various categorical null value scenarios:")
print("=" * 60)

for i, cat_list in enumerate(test_cases):
    df = pd.DataFrame({"col": pd.Categorical(cat_list)})
    result = from_dataframe(df.__dataframe__())

    print(f"\nTest case {i+1}: {cat_list}")
    print(f"Original: {df['col'].tolist()}")
    print(f"Result:   {result['col'].tolist()}")

    # Check if nulls are preserved
    orig_nulls = [i for i, v in enumerate(df['col'].isna()) if v]
    result_nulls = [i for i, v in enumerate(result['col'].isna()) if v]

    if orig_nulls != result_nulls:
        print(f"❌ NULL PRESERVATION FAILED!")
        print(f"   Original null indices: {orig_nulls}")
        print(f"   Result null indices: {result_nulls}")
    else:
        print(f"✓ Nulls preserved correctly")

# Now let's look at the internal representation
print("\n" + "=" * 60)
print("Examining internal representation:")
print("=" * 60)

df = pd.DataFrame({"col": pd.Categorical(["cat1", None, "cat2"])})
print(f"\nOriginal categorical:")
print(f"  Values: {df['col'].tolist()}")
print(f"  Codes: {df['col'].cat.codes.tolist()}")
print(f"  Categories: {df['col'].cat.categories.tolist()}")

# Get interchange object
interchange_obj = df.__dataframe__()
col = interchange_obj.get_column_by_name("col")

# Check null description
null_kind, sentinel_val = col.describe_null
print(f"\nInterchange null representation:")
print(f"  Null kind: {null_kind} ({null_kind.name if hasattr(null_kind, 'name') else ''})")
print(f"  Sentinel value: {sentinel_val}")

# Get the buffers
buffers = col.get_buffers()
codes_buff, codes_dtype = buffers["data"]

# Access the codes directly
import ctypes
import numpy as np

# Based on the code in from_dataframe.py, reconstruct the codes array
from pandas.core.interchange.from_dataframe import buffer_to_ndarray

codes = buffer_to_ndarray(
    codes_buff, codes_dtype, offset=col.offset, length=col.size()
)
print(f"\nCodes from interchange buffer: {codes}")

# Now simulate what the problematic line does
categories = np.array(["cat1", "cat2"])
print(f"\nSimulating the bug:")
print(f"  categories = {categories}")
print(f"  codes = {codes}")
print(f"  codes % len(categories) = {codes % len(categories)}")
print(f"  Result after modulo: {categories[codes % len(categories)]}")
print(f"  Expected: ['cat1', None, 'cat2'] but got: {categories[codes % len(categories)].tolist()}")