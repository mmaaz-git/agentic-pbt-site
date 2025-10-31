#!/usr/bin/env python3
"""Test to understand if HB format can theoretically represent empty matrices - fixed."""

import scipy.sparse
import scipy.io
import tempfile
import os
import numpy as np

print("Testing if HB format can represent empty matrices")
print("=" * 60)

# Create a properly formatted HB file for an empty 3x3 matrix
# Each line must have the correct character count
hb_content = """Empty sparse matrix test                                                01234567
             3             1             0             0             0
RSA                     3             3             0             0
(4I14)         (0I14)         (0E25.16)
             1             1             1             1
"""

fd, path = tempfile.mkstemp(suffix='.hb')
try:
    with os.fdopen(fd, 'w') as f:
        f.write(hb_content)

    print(f"Created test HB file for empty matrix: {path}")

    # Try to read it back
    try:
        matrix = scipy.io.hb_read(path)
        print(f"Successfully read empty matrix!")
        print(f"  Shape: {matrix.shape}")
        print(f"  Non-zero elements: {matrix.nnz}")
        print(f"  Type: {type(matrix)}")

        # Test if it's really empty
        if matrix.nnz == 0:
            print("  Confirmed: Matrix is empty (nnz = 0)")

    except Exception as e:
        print(f"Failed to read: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

finally:
    if os.path.exists(path):
        os.unlink(path)

print("\n" + "=" * 60)
print("Conclusion: Can HB format represent empty matrices?")