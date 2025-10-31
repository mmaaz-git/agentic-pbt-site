import pandas.errors
import pytest

# Test with the specific failing input
invalid_methodtype = "invalid"

with pytest.raises(ValueError) as exc_info:
    pandas.errors.AbstractMethodError(object(), methodtype=invalid_methodtype)

error_msg = str(exc_info.value)
valid_types = {"method", "classmethod", "staticmethod", "property"}

print(f"Error message: {error_msg}")
print(f"Checking if valid types appear in 'got' part...")

parts = error_msg.split(", got")
if len(parts) == 2:
    print(f"Part before ', got': {parts[0]}")
    print(f"Part after ', got': {parts[1]}")
    for valid_type in valid_types:
        if valid_type in parts[1]:
            print(f"ERROR: Valid type '{valid_type}' appears in 'got X' part - this is backwards!")
        else:
            print(f"OK: Valid type '{valid_type}' does not appear in 'got X' part")