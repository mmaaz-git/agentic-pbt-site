"""Verify the bug exists in the actual dask code"""
from multiprocessing import current_process
import psutil

# Simulate what happens in dask's _Tracker.run() method
print("=== Simulating dask's _Tracker.run() behavior ===\n")

# Line 261 in dask code
pid = current_process()
print(f"Line 261: pid = current_process()")
print(f"  pid type: {type(pid)}")
print(f"  pid value: {pid}")
print(f"  pid.pid: {pid.pid}")

# Mock parent process for testing
parent = psutil.Process()
print(f"\nParent process PID: {parent.pid}")

# Get children processes
children = parent.children()
print(f"Children processes: {[p.pid for p in children]}")

# Line 252: The buggy comparison
print("\n=== Testing the buggy comparison (line 252) ===")
for p in children[:3] if children else []:  # Test first 3 children if any
    print(f"  p.pid ({p.pid}, type: {type(p.pid)}) != pid ({pid}, type: {type(pid)})")
    print(f"  Result: {p.pid != pid}")
    print(f"  This will ALWAYS be True because int != Process object")
    print()

# What the comparison should be
print("=== Testing the correct comparison ===")
for p in children[:3] if children else []:
    print(f"  p.pid ({p.pid}) != pid.pid ({pid.pid})")
    print(f"  Result: {p.pid != pid.pid}")
    print(f"  This correctly compares two integers")
    print()

# The actual issue
print("=== The Issue ===")
print(f"The bug: comparing p.pid (int) with pid (Process object)")
print(f"  p.pid != pid will ALWAYS be True")
print(f"  This means the tracker process is never filtered out")
print(f"\nThe fix: compare p.pid with pid.pid (both integers)")
print(f"  p.pid != pid.pid correctly filters when PIDs match")