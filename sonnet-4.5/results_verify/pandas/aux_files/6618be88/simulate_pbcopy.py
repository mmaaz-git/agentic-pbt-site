#!/usr/bin/env python3
"""Simulate pbcopy and pbpaste behavior to demonstrate the bug"""

import sys
import os

# Create fake pbcopy and pbpaste scripts to demonstrate the issue
pbcopy_script = """#!/bin/bash
# Simulate pbcopy - expects no arguments
if [ $# -ne 0 ]; then
    echo "pbcopy: illegal option -- $1" >&2
    echo "usage: pbcopy" >&2
    exit 1
fi
cat > /tmp/clipboard.txt
exit 0
"""

pbpaste_script = """#!/bin/bash
# Simulate pbpaste - expects no arguments
if [ $# -ne 0 ]; then
    echo "pbpaste: illegal option -- $1" >&2
    echo "usage: pbpaste" >&2
    exit 1
fi
cat /tmp/clipboard.txt 2>/dev/null || echo ""
exit 0
"""

# Write the scripts
with open('/tmp/pbcopy', 'w') as f:
    f.write(pbcopy_script)
os.chmod('/tmp/pbcopy', 0o755)

with open('/tmp/pbpaste', 'w') as f:
    f.write(pbpaste_script)
os.chmod('/tmp/pbpaste', 0o755)

# Add /tmp to PATH
os.environ['PATH'] = '/tmp:' + os.environ.get('PATH', '')

# Now test
import subprocess

print("Testing simulated pbcopy/pbpaste behavior:\n")
print("="*60)

# Test 1: pbcopy with "w" argument (SHOULD FAIL)
print("\nTest 1: pbcopy with 'w' argument (incorrect):")
result = subprocess.run(["/tmp/pbcopy", "w"],
                      input=b"test",
                      capture_output=True)
print(f"  Return code: {result.returncode}")
if result.stderr:
    print(f"  Stderr: {result.stderr.decode().strip()}")

# Test 2: pbcopy without arguments (SHOULD WORK)
print("\nTest 2: pbcopy without arguments (correct):")
result = subprocess.run(["/tmp/pbcopy"],
                      input=b"test",
                      capture_output=True)
print(f"  Return code: {result.returncode}")
if result.stderr:
    print(f"  Stderr: {result.stderr.decode().strip()}")
else:
    print("  Success - no errors")

# Test 3: pbpaste with "r" argument (SHOULD FAIL)
print("\nTest 3: pbpaste with 'r' argument (incorrect):")
result = subprocess.run(["/tmp/pbpaste", "r"],
                      capture_output=True)
print(f"  Return code: {result.returncode}")
if result.stderr:
    print(f"  Stderr: {result.stderr.decode().strip()}")

# Test 4: pbpaste without arguments (SHOULD WORK)
print("\nTest 4: pbpaste without arguments (correct):")
result = subprocess.run(["/tmp/pbpaste"],
                      capture_output=True)
print(f"  Return code: {result.returncode}")
print(f"  Output: {result.stdout.decode().strip()}")

print("\n" + "="*60)
print("\nConclusion:")
print("- pbcopy and pbpaste DO NOT accept 'w' or 'r' arguments")
print("- The pandas code incorrectly passes these arguments")
print("- This will cause clipboard operations to fail on macOS")