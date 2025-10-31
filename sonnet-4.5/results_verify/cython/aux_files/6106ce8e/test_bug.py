#!/usr/bin/env python3
"""Test script to reproduce the chars_to_ranges bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Plex.Regexps import chars_to_ranges, Any

print("Testing chars_to_ranges function")
print("=" * 50)

# Test case 1: duplicate characters '00'
print("\nTest 1: Input '00' (duplicate zeros)")
ranges = chars_to_ranges('00')
print(f"Ranges returned: {ranges}")

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input characters: {set('00')}")
print(f"Covered characters: {covered}")
print(f"Extra characters included: {covered - set('00')}")

# Test case 2: duplicate characters 'aaa'
print("\nTest 2: Input 'aaa' (triple a)")
ranges = chars_to_ranges('aaa')
print(f"Ranges returned: {ranges}")

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input characters: {set('aaa')}")
print(f"Covered characters: {covered}")
print(f"Extra characters included: {covered - set('aaa')}")

# Test case 3: duplicate mixed 'aabbcc'
print("\nTest 3: Input 'aabbcc' (duplicate pairs)")
ranges = chars_to_ranges('aabbcc')
print(f"Ranges returned: {ranges}")

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input characters: {set('aabbcc')}")
print(f"Covered characters: {covered}")
print(f"Extra characters included: {covered - set('aabbcc')}")

# Test case 4: no duplicates 'abc'
print("\nTest 4: Input 'abc' (no duplicates)")
ranges = chars_to_ranges('abc')
print(f"Ranges returned: {ranges}")

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input characters: {set('abc')}")
print(f"Covered characters: {covered}")
print(f"Extra characters included: {covered - set('abc')}")

# Test case 5: single character '0'
print("\nTest 5: Input '0' (single character)")
ranges = chars_to_ranges('0')
print(f"Ranges returned: {ranges}")

covered = set()
for i in range(0, len(ranges), 2):
    for code in range(ranges[i], ranges[i+1]):
        covered.add(chr(code))

print(f"Input characters: {set('0')}")
print(f"Covered characters: {covered}")
print(f"Extra characters included: {covered - set('0')}")

print("\n" + "=" * 50)
print("Summary:")
print("The function incorrectly handles duplicates, extending ranges beyond the input characters.")