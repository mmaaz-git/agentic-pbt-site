from django.db.backends.sqlite3._functions import _sqlite_lpad, _sqlite_rpad

# Test case from the bug report
text = "hello"
length = 10
fill_text = ""

print("Testing LPAD and RPAD with empty fill_text")
print("=" * 50)
print(f"Input: text='{text}', length={length}, fill_text='{fill_text}'")
print()

# Test LPAD
result_lpad = _sqlite_lpad(text, length, fill_text)
print(f"LPAD result: {result_lpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_lpad)}")
print()

# Test RPAD
result_rpad = _sqlite_rpad(text, length, fill_text)
print(f"RPAD result: {result_rpad!r}")
print(f"Expected length: {length}, Actual length: {len(result_rpad)}")
print()

# Test edge cases
print("Additional edge cases:")
print("-" * 30)

# Case 1: Text longer than target length with empty padding
longer_text = "verylongtext"
target_length = 5
result_lpad_truncate = _sqlite_lpad(longer_text, target_length, fill_text)
result_rpad_truncate = _sqlite_rpad(longer_text, target_length, fill_text)
print(f"Text longer than target ('{longer_text}', length={target_length}):")
print(f"  LPAD: {result_lpad_truncate!r} (length: {len(result_lpad_truncate)})")
print(f"  RPAD: {result_rpad_truncate!r} (length: {len(result_rpad_truncate)})")
print()

# Case 2: Normal case with non-empty padding (for comparison)
normal_fill = "x"
result_lpad_normal = _sqlite_lpad(text, length, normal_fill)
result_rpad_normal = _sqlite_rpad(text, length, normal_fill)
print(f"Normal case with fill_text='{normal_fill}':")
print(f"  LPAD: {result_lpad_normal!r} (length: {len(result_lpad_normal)})")
print(f"  RPAD: {result_rpad_normal!r} (length: {len(result_rpad_normal)})")
print()

# Case 3: NULL/None handling
result_lpad_none = _sqlite_lpad(text, length, None)
result_rpad_none = _sqlite_rpad(text, length, None)
print(f"With fill_text=None:")
print(f"  LPAD: {result_lpad_none!r}")
print(f"  RPAD: {result_rpad_none!r}")