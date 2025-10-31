#!/usr/bin/env python3

import sys
import traceback
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import resolve_depend

print("Attempting to call resolve_depend with empty string:")
try:
    result = resolve_depend("", ())
except IndexError:
    traceback.print_exc()

print("\nExamining the problematic line:")
depend = ""
print(f"depend = '{depend}'")
print(f"len(depend) = {len(depend)}")
print(f"bool(depend) = {bool(depend)}")
print("Trying to access depend[0]...")
try:
    first_char = depend[0]
    print(f"depend[0] = '{first_char}'")
except IndexError as e:
    print(f"IndexError on depend[0]: {e}")

print("Trying to access depend[-1]...")
try:
    last_char = depend[-1]
    print(f"depend[-1] = '{last_char}'")
except IndexError as e:
    print(f"IndexError on depend[-1]: {e}")