#!/usr/bin/env python3
"""Test Python's standard slicing behavior when start > stop"""

test_strings = ["hello", "0", "abc", "x"]

print("Testing Python's standard slicing behavior when start > stop:")
print("=" * 60)

for s in test_strings:
    for start, stop in [(3, 1), (1, 0), (5, 2), (2, 2)]:
        slice_result = s[start:stop]
        manual_concat = s[:start] + "REPL" + s[stop:]
        print(f"String: {s!r}")
        print(f"  s[{start}:{stop}] = {slice_result!r}")
        print(f"  s[:{start}] + 'REPL' + s[{stop}:] = {manual_concat!r}")
        print()

print("\nKey observations:")
print("1. When start >= stop, s[start:stop] returns empty string ''")
print("2. s[:start] + repl + s[stop:] still works correctly")
print("3. This means when start > stop, we insert at position 'start' and continue from position 'stop'")