from llm.utils import remove_dict_none_values

# Test case 1: Basic inconsistent handling
d1 = {"direct": {"x": None}, "in_list": [{"y": None}]}
result1 = remove_dict_none_values(d1)

print("Test case 1: Basic inconsistent handling")
print(f"Input:  {d1}")
print(f"Output: {result1}")
print()

# Test case 2: None values in lists are not removed
d2 = {"a": [1, None, 2]}
result2 = remove_dict_none_values(d2)

print("Test case 2: None values in lists are not removed")
print(f"Input:  {d2}")
print(f"Output: {result2}")
print()

# Test case 3: Empty dicts in lists are not removed
d3 = {"a": [{"b": None}]}
result3 = remove_dict_none_values(d3)

print("Test case 3: Empty dicts in lists are not removed")
print(f"Input:  {d3}")
print(f"Output: {result3}")
print()

# Test case 4: Direct nested dict with None is fully removed
d4 = {"direct": {"nested": {"all_none": None}}}
result4 = remove_dict_none_values(d4)

print("Test case 4: Direct nested dict with None is fully removed")
print(f"Input:  {d4}")
print(f"Output: {result4}")