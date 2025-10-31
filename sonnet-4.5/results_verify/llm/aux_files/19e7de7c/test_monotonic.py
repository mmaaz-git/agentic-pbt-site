#!/usr/bin/env python3
import sys
sys.path.insert(0, '../../envs/llm_env/lib/python3.13/site-packages')

import time
from llm.utils import monotonic_ulid

# Test 1: Normal case - monotonicity with forward time
print("Test 1: Normal forward time monotonicity")
ulids = []
for i in range(5):
    ulid = monotonic_ulid()
    ulids.append(ulid)
    print(f"  ULID {i+1}: {ulid}")
    if i > 0:
        assert ulid > ulids[i-1], f"Monotonicity violated: {ulid} <= {ulids[i-1]}"
print("  ✓ All ULIDs are strictly increasing\n")

# Test 2: Examine the implementation
print("Test 2: Examining the implementation logic")
import llm.utils
import inspect
source = inspect.getsource(llm.utils.monotonic_ulid)
print("Key logic points:")
lines = source.split('\n')
for i, line in enumerate(lines):
    if 'now_ms == last_ms' in line:
        print(f"  Line {i}: Same millisecond handling found")
    if 'now_ms < last_ms' in line:
        print(f"  Line {i}: Backward clock handling found")
    if 'New millisecond' in line:
        print(f"  Line {i}: Forward time handling found")

# Test 3: Simulate the bug scenario (without actually changing system time)
print("\nTest 3: Simulating clock skew scenario")
print("  Without changing system time, let's trace through the logic:")

# Get first ULID
ulid1 = monotonic_ulid()
print(f"  First ULID: {ulid1}")
print(f"  Timestamp (ms): {ulid1.milliseconds}")
print(f"  As bytes: {ulid1.bytes.hex()}")

# Get second ULID
ulid2 = monotonic_ulid()
print(f"\n  Second ULID: {ulid2}")
print(f"  Timestamp (ms): {ulid2.milliseconds}")
print(f"  As bytes: {ulid2.bytes.hex()}")

print("\n  Analysis of what would happen if clock went backwards:")
print("  - If system clock went from T=1000ms to T=999ms")
print("  - The code would hit the 'New millisecond, start fresh' branch")
print("  - A fresh ULID with timestamp 999 would be generated")
print("  - This would be LESS than the previous ULID with timestamp 1000")
print("  - Violating the strict monotonicity guarantee")

# Test 4: Check for clock backward handling in the code
print("\nTest 4: Checking for backward clock handling")
with open('../../envs/llm_env/lib/python3.13/site-packages/llm/utils.py', 'r') as f:
    content = f.read()

# Look for the critical section
if 'if now_ms < last_ms:' in content:
    print("  ✓ Code explicitly handles backward clock movement")
else:
    print("  ✗ Code does NOT handle backward clock movement")

# Check the actual logic flow
func_start = content.find('def monotonic_ulid()')
func_end = content.find('\ndef ', func_start + 1)
func_code = content[func_start:func_end]

has_same_ms_check = 'if now_ms == last_ms:' in func_code
has_backward_check = 'if now_ms < last_ms:' in func_code
has_else_new_ms = '# New millisecond, start fresh' in func_code

print(f"  Has same millisecond check: {has_same_ms_check}")
print(f"  Has backward clock check: {has_backward_check}")
print(f"  Has else clause for new millisecond: {has_else_new_ms}")

if has_same_ms_check and not has_backward_check and has_else_new_ms:
    print("\n  CONFIRMED: The bug exists - backward clock movement is not handled!")
    print("  When now_ms < last_ms, the code falls through to 'New millisecond'")
    print("  This generates a ULID with an earlier timestamp, violating monotonicity")