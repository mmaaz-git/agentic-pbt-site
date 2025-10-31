from xarray.core.formatting_html import _load_static_files

first_call = _load_static_files()
print(f"First call: {len(first_call)} items")

first_call.append("INJECTED")

second_call = _load_static_files()
print(f"Second call: {len(second_call)} items")
print(f"Same object: {first_call is second_call}")
print(f"Last item: {second_call[-1]}")