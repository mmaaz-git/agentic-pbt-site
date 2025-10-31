from Cython.TestUtils import _parse_pattern

print("Case 1: Start marker without closing slash")
try:
    result = _parse_pattern("/start")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 2: Single slash")
try:
    result = _parse_pattern("/")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 3: End marker without closing slash")
try:
    result = _parse_pattern(":/end")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 4: Escaped slash without closing slash")
try:
    result = _parse_pattern("/\\/escaped")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 5: Valid pattern with closing slash")
try:
    result = _parse_pattern("/start/pattern")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")

print("\nCase 6: Valid pattern with end marker")
try:
    result = _parse_pattern("/start/:/end/pattern")
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError: {e}")