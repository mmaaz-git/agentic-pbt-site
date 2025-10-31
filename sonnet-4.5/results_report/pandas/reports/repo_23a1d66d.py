from pandas.api.types import infer_dtype

# Test with pure floats
pure_floats = [1.5, 2.5, 3.5]
result_pure = infer_dtype(pure_floats, skipna=True)
print(f"infer_dtype([1.5, 2.5, 3.5], skipna=True) = '{result_pure}'")

# Test with floats containing None
floats_with_none = [1.5, None, 3.5]
result_with_none = infer_dtype(floats_with_none, skipna=True)
print(f"infer_dtype([1.5, None, 3.5], skipna=True) = '{result_with_none}'")

# Test with single float
single_float = [0.0]
result_single = infer_dtype(single_float, skipna=True)
print(f"infer_dtype([0.0], skipna=True) = '{result_single}'")

# Test with single float and None
single_float_with_none = [0.0, None]
result_single_with_none = infer_dtype(single_float_with_none, skipna=True)
print(f"infer_dtype([0.0, None], skipna=True) = '{result_single_with_none}'")

# Assert that results should be the same when skipna=True
print("\nAssertion check:")
assert result_pure == result_with_none, \
    f"Expected both to return '{result_pure}', but got '{result_with_none}' for floats_with_none"