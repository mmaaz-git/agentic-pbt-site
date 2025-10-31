#!/usr/bin/env python3
"""Test the reported bug in Django's autocomplete functionality."""

import os
import sys

print("Testing Django autocomplete bug report")
print("=" * 50)

# Test 1: Reproduce the exact bug scenario
print("\nTest 1: Reproducing the reported bug")
print("-" * 40)

cwords = ["django-admin", "migrate"]
cword = 0

print(f"cwords = {cwords}")
print(f"cword = {cword}")
print(f"Attempting: cwords[cword - 1] = cwords[{cword - 1}]")

try:
    curr = cwords[cword - 1]
    print(f"Result: curr = {curr!r}")
except IndexError as e:
    curr = ""
    print(f"IndexError caught: {e}")
    print(f"Result: curr = {curr!r}")

print(f"\nExpected according to bug report: ''")
print(f"Actual result: {curr!r}")

if cword == 0 and len(cwords) > 0:
    if curr == "":
        print("✓ Behavior matches expectation (no bug)")
    else:
        print(f"✗ Bug confirmed: When cword=0, expected curr='', but got {curr!r}")

# Test 2: Check Python's negative indexing behavior
print("\n\nTest 2: Understanding Python's negative indexing")
print("-" * 40)

test_list = ["first", "second", "third"]
print(f"test_list = {test_list}")

for index in [0, -1, -2, -3]:
    try:
        value = test_list[index]
        print(f"test_list[{index:2d}] = {value!r}")
    except IndexError as e:
        print(f"test_list[{index:2d}] raises IndexError: {e}")

# Test 3: Check what COMP_CWORD actually means
print("\n\nTest 3: Understanding COMP_CWORD behavior")
print("-" * 40)
print("In bash completion:")
print("- COMP_WORDS is an array of words in the command line")
print("- COMP_CWORD is the index of the current word being completed")
print("\nFor 'django-admin <TAB>':")
print("  COMP_WORDS=['django-admin']")
print("  COMP_CWORD should be 1 (pointing to position after 'django-admin')")
print("\nFor 'django-admin migrate <TAB>':")
print("  COMP_WORDS=['django-admin', 'migrate']")
print("  COMP_CWORD should be 2 (pointing to position after 'migrate')")

# Test 4: Test with actual bash behavior simulation
print("\n\nTest 4: Simulating actual bash completion behavior")
print("-" * 40)

def test_django_autocomplete(comp_words_str, comp_cword):
    """Simulate Django's autocomplete behavior."""
    # Django splits on the first element
    cwords = comp_words_str.split()[1:]  # Skip program name

    print(f"COMP_WORDS='{comp_words_str}'")
    print(f"COMP_CWORD={comp_cword}")
    print(f"cwords (after split[1:]) = {cwords}")
    print(f"Adjusted cword for indexing = {comp_cword}")

    try:
        curr = cwords[comp_cword - 1]
        print(f"curr = cwords[{comp_cword - 1}] = {curr!r}")
    except IndexError:
        curr = ""
        print(f"IndexError: curr = ''")

    print(f"Result: curr = {curr!r}")
    print()
    return curr

# Simulate different completion scenarios
scenarios = [
    ("django-admin", 1),  # Just typed django-admin and hit TAB
    ("django-admin migrate", 2),  # Typed django-admin migrate and hit TAB
    ("django-admin migrate --database", 3),  # More arguments
]

for words, cword in scenarios:
    result = test_django_autocomplete(words, cword)

# Test 5: Check if bug exists with COMP_CWORD=0
print("\nTest 5: Edge case with COMP_CWORD=0")
print("-" * 40)
print("Note: COMP_CWORD=0 would mean we're completing the program name itself")
print("This is unlikely in practice as bash completion is triggered after the program name")

comp_words_str = "django-admin"
comp_cword = 0
cwords = comp_words_str.split()[1:]  # This would be empty list []

print(f"COMP_WORDS='{comp_words_str}'")
print(f"COMP_CWORD={comp_cword}")
print(f"cwords (after split[1:]) = {cwords}")

try:
    curr = cwords[comp_cword - 1]  # cwords[-1] on empty list
    print(f"curr = cwords[{comp_cword - 1}] = {curr!r}")
except IndexError:
    curr = ""
    print(f"IndexError caught: curr = ''")

print(f"Result: curr = {curr!r}")