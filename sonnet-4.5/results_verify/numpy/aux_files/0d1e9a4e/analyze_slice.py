#!/usr/bin/env python3
"""Analyze the slicing behavior that causes the bug"""

# Simulate the function's behavior
COMMON_WORDS = (
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipisicing",
    "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
    "et", "dolore", "magna", "aliqua",
)

print(f"Total COMMON_WORDS: {len(COMMON_WORDS)} words")
print()

# The issue is in line 285: word_list = word_list[:count]
# When count is negative, Python's list slicing has specific behavior

word_list = list(COMMON_WORDS)

for count in [-1, -5, -10, -19, -20, -100]:
    sliced = word_list[:count]
    print(f"word_list[:{count}] returns {len(sliced)} words")
    if count == -1:
        print(f"  Explanation: [:-1] means 'all but the last 1 element'")
        print(f"  Result: {sliced}")
    elif count == -5:
        print(f"  Explanation: [:-5] means 'all but the last 5 elements'")

print()
print("Python list slicing with negative indices:")
print("  lst[:n] where n > 0: take first n elements")
print("  lst[:n] where n < 0: take all but the last |n| elements")
print("  lst[:0]: empty list")
print("  lst[:-len(lst)]: all but last len(lst) elements = empty")
print("  lst[:-len(lst)-1]: still empty (negative index beyond start)")