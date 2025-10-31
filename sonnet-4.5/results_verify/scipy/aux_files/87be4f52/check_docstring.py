#!/usr/bin/env python3
"""Check the docstring of Rotation.mean method."""

from scipy.spatial.transform import Rotation
import inspect

print("="*60)
print("ROTATION.MEAN DOCSTRING")
print("="*60)

# Get the docstring
if hasattr(Rotation.mean, '__doc__'):
    print(Rotation.mean.__doc__)
else:
    print("No docstring found")

print("\n" + "="*60)
print("ROTATION CLASS DOCSTRING (relevant parts)")
print("="*60)

# Check if there's info about mean in the class docstring
if hasattr(Rotation, '__doc__'):
    doc_lines = Rotation.__doc__.split('\n')
    for i, line in enumerate(doc_lines):
        if 'mean' in line.lower():
            # Print context around mentions of 'mean'
            start = max(0, i-2)
            end = min(len(doc_lines), i+3)
            for j in range(start, end):
                print(doc_lines[j])
            print("...")

print("\n" + "="*60)
print("METHOD SIGNATURE")
print("="*60)

# Get method signature
try:
    sig = inspect.signature(Rotation.mean)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Could not get signature: {e}")

# Try to get more info
print("\n" + "="*60)
print("ADDITIONAL INFO")
print("="*60)

print(f"Type: {type(Rotation.mean)}")
print(f"Module: {Rotation.mean.__module__ if hasattr(Rotation.mean, '__module__') else 'N/A'}")