#!/usr/bin/env python3
"""Check Python's hexversion format to understand the standard."""

import sys

hexver = sys.hexversion
print(f"Python sys.hexversion: {hex(hexver)}")
print(f"Python sys.version_info: {sys.version_info}")

# Extract components using bit shifting
major = (hexver >> 24) & 0xFF
minor = (hexver >> 16) & 0xFF
micro = (hexver >> 8) & 0xFF
level = (hexver >> 4) & 0xF
serial = hexver & 0xF

print(f"\nExtracted from hexversion:")
print(f"  Major: {major}")
print(f"  Minor: {minor}")
print(f"  Micro: {micro}")
print(f"  Level: {level:x} ({level})")
print(f"  Serial: {serial}")

# Map release level
level_map = {0xa: 'alpha', 0xb: 'beta', 0xc: 'candidate', 0xf: 'final'}
level_name = level_map.get(level, f'unknown({level})')

print(f"\nComparison:")
print(f"  Major: {major} == {sys.version_info.major}: {major == sys.version_info.major}")
print(f"  Minor: {minor} == {sys.version_info.minor}: {minor == sys.version_info.minor}")
print(f"  Micro: {micro} == {sys.version_info.micro}: {micro == sys.version_info.micro}")
print(f"  Level: '{level_name}' == '{sys.version_info.releaselevel}': {level_name == sys.version_info.releaselevel}")
print(f"  Serial: {serial} == {sys.version_info.serial}: {serial == sys.version_info.serial}")

# Test Cython's py_version_hex with valid Python version values
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Compiler.Naming import py_version_hex

# Test with current Python version
level_value = {'alpha': 0xa, 'beta': 0xb, 'candidate': 0xc, 'final': 0xf}[sys.version_info.releaselevel]
cython_hex = py_version_hex(
    sys.version_info.major,
    sys.version_info.minor,
    sys.version_info.micro,
    level_value,
    sys.version_info.serial
)

print(f"\nCython py_version_hex with current Python version:")
print(f"  Result: {hex(cython_hex)}")
print(f"  Matches sys.hexversion: {cython_hex == sys.hexversion}")