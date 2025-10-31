#!/usr/bin/env python3
"""Test to understand if HB format can theoretically represent empty matrices - proper format."""

import scipy.sparse
import scipy.io
import tempfile
import os

print("Testing if HB format can represent empty matrices")
print("=" * 60)

# Create a properly formatted HB file for an empty 3x3 matrix
# Line 1: Must be >72 chars total (72 for title, rest for key)
# Line 2: 5 fields of 14 chars each = 70 chars (right justified integers)
# Line 3: Type(3) + space(11) + 4 fields of 14 chars = 70 chars
# Line 4: 3 format specifications of 16 chars each
hb_content = """Empty sparse matrix test                                                01234567
             3             1             0             0             0
RSA                     3             3             0             0
(4I14)          (0I14)          (0E25.16)
             1             1             1             1
"""

# Ensure proper line lengths
lines = hb_content.strip().split('\n')
print(f"Line 1 length: {len(lines[0])} (should be >72)")
print(f"Line 2 length: {len(lines[1])} (should be >=56)")
print(f"Line 3 length: {len(lines[2])} (should be >=70)")

# Pad line 3 to ensure it's at least 70 chars
if len(lines[2]) < 70:
    lines[2] = lines[2].ljust(70)

hb_content_fixed = '\n'.join(lines) + '\n'

fd, path = tempfile.mkstemp(suffix='.hb')
try:
    with os.fdopen(fd, 'w') as f:
        f.write(hb_content_fixed)

    print(f"\nCreated test HB file for empty matrix")

    # Try to read it back
    try:
        matrix = scipy.io.hb_read(path)
        print(f"Successfully read empty matrix!")
        print(f"  Shape: {matrix.shape}")
        print(f"  Non-zero elements: {matrix.nnz}")
        print(f"  Data: {matrix.data}")
        print("\nCONCLUSION: HB format CAN represent empty matrices!")

    except Exception as e:
        print(f"Failed to read: {type(e).__name__}: {e}")

finally:
    if os.path.exists(path):
        os.unlink(path)