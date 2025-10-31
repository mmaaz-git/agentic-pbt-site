#!/usr/bin/env python3
"""Examine the outputs of pylong_join functions"""

from Cython.Utility import pylong_join, _pylong_join

print("Testing the outputs for various count values:\n")

for count in range(5):
    pub = pylong_join(count)
    priv = _pylong_join(count)
    print(f"count={count}:")
    print(f"  pylong_join:  {repr(pub)}")
    print(f"  _pylong_join: {repr(priv)}")
    print()

# Test with different parameters
print("Testing with custom parameters:")
pub = pylong_join(2, digits_ptr='my_digits', join_type='uint64_t')
priv = _pylong_join(2, digits_ptr='my_digits', join_type='uint64_t')
print(f"pylong_join(2, 'my_digits', 'uint64_t'):")
print(f"  pylong_join:  {repr(pub)}")
print(f"  _pylong_join: {repr(priv)}")