#!/usr/bin/env python3
"""
Minimal reproduction of the Cython debugger CyBreak.complete bug.
This demonstrates the issue with text[:-0] returning an empty string.
"""

text = "cy break spam "
word = ""

print("Input:")
print(f"  text = {repr(text)}")
print(f"  word = {repr(word)}")
print()

# The bug: when word is empty, text[:-len(word)] becomes text[:-0]
# which in Python returns an empty string, not the original string
slice_result = text[:-len(word)]
print("Slicing operation:")
print(f"  len(word) = {len(word)}")
print(f"  text[:-len(word)] = text[:-{len(word)}] = {repr(slice_result)}")
print()

# This causes seen to be empty when it should contain the already-typed function names
seen = set(slice_result.split())
print("Result of splitting:")
print(f"  seen = set(text[:-len(word)].split()) = {seen}")
print()

# Simulate the completion logic
all_names = ["spam", "eggs", "ham", "bacon"]
result = [n for n in all_names if n.startswith(word) and n not in seen]
print("Completion result:")
print(f"  all_names = {all_names}")
print(f"  result = [n for n in all_names if n.startswith(word) and n not in seen]")
print(f"  result = {result}")
print()

print("Bug manifestation:")
print(f"  'spam' in result = {'spam' in result}")
print(f"  Expected: False (spam should be filtered out as it's already typed)")
print(f"  Actual: True (spam appears in suggestions due to empty seen set)")