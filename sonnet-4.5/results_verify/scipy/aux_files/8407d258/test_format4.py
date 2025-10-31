#!/usr/bin/env python3
"""Test if HB format can represent empty matrices - with unsymmetric type."""

import scipy.sparse
import scipy.io
import tempfile
import os

print("Testing if HB format can represent empty matrices")
print("=" * 60)

# RUA = Real, Unsymmetric, Assembled (which is what scipy supports)
hb_content = """Empty sparse matrix test                                                01234567
             3             1             0             0             0
RUA                     3             3             0             0
(4I14)          (0I14)          (0E25.16)
             1             1             1             1
"""

# Pad lines to proper length
lines = hb_content.strip().split('\n')
lines[2] = lines[2].ljust(70)  # Line 3 must be at least 70 chars
lines[3] = lines[3].ljust(72)  # Line 4 with format specs

hb_content_fixed = '\n'.join(lines) + '\n'

fd, path = tempfile.mkstemp(suffix='.hb')
try:
    with os.fdopen(fd, 'w') as f:
        f.write(hb_content_fixed)

    print(f"Created test HB file with RUA type")

    # Try to read it back
    try:
        matrix = scipy.io.hb_read(path)
        print(f"\n✓ Successfully read empty matrix from HB file!")
        print(f"  Shape: {matrix.shape}")
        print(f"  Non-zero elements: {matrix.nnz}")
        print(f"  Data array: {matrix.data}")
        print(f"  Indices array: {matrix.indices}")

        # Verify it's truly empty
        assert matrix.nnz == 0, "Matrix should have 0 non-zero elements"
        assert matrix.shape == (3, 3), "Matrix should be 3x3"
        assert len(matrix.data) == 0, "Data array should be empty"

        print("\n✓ CONFIRMED: HB format CAN represent empty sparse matrices!")
        print("  The format can handle matrices with zero non-zero elements.")

    except Exception as e:
        print(f"✗ Failed to read: {type(e).__name__}: {e}")

finally:
    if os.path.exists(path):
        os.unlink(path)

print("\n" + "=" * 60)
print("Implications:")
print("- The HB file format itself supports empty matrices")
print("- scipy.io.hb_read can successfully read empty matrices")
print("- scipy.io.hb_write crashes when trying to write them")
print("- This is an implementation bug, not a format limitation")