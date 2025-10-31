#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import parse_list

# Test the reported failing input
print("Testing parse_list with incomplete quote: [']")
try:
    result = parse_list("[']")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting parse_list with incomplete quote: [\"]")
try:
    result = parse_list('[""]')
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting parse_list with valid inputs:")
print(f"parse_list(''): {parse_list('')}")
print(f"parse_list('a'): {parse_list('a')}")
print(f"parse_list('a b c'): {parse_list('a b c')}")
print(f"parse_list('[a, b, c]'): {parse_list('[a, b, c]')}")
print('parse_list(\'[a, ",a", "a,", ","]\'): ', parse_list('[a, ",a", "a,", ","]'))