from xarray.core.formatting_html import _load_static_files

# First call to _load_static_files()
first_call = _load_static_files()
print(f"First call: {len(first_call)} items")
print(f"Type: {type(first_call)}")

# Mutate the cached list by appending an item
first_call.append("INJECTED")
print(f"After mutation: {len(first_call)} items")

# Second call to _load_static_files()
second_call = _load_static_files()
print(f"Second call: {len(second_call)} items")

# Check if they're the same object
print(f"Same object: {first_call is second_call}")

# Show the last item to prove mutation persisted
print(f"Last item: {repr(second_call[-1])}")

# Third call to verify permanence
third_call = _load_static_files()
print(f"Third call: {len(third_call)} items")