#!/usr/bin/env python3
"""Test script to reproduce the LegacyVersion.public bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas.util.version

# First test: Version class (PEP 440 compliant)
print("=" * 60)
print("Testing Version class (PEP 440 compliant):")
print("=" * 60)

v1 = pandas.util.version.parse("1.0.0+local")
print(f"Input: '1.0.0+local'")
print(f"Type: {type(v1).__name__}")
print(f"Version.public: '{v1.public}'")
print(f"Has '+': {'+' in v1.public}")
print()

v1b = pandas.util.version.parse("2.1.0rc1+git.abc123")
print(f"Input: '2.1.0rc1+git.abc123'")
print(f"Type: {type(v1b).__name__}")
print(f"Version.public: '{v1b.public}'")
print(f"Has '+': {'+' in v1b.public}")
print()

# Second test: LegacyVersion class (non-PEP 440)
print("=" * 60)
print("Testing LegacyVersion class (non-PEP 440):")
print("=" * 60)

v2 = pandas.util.version.parse("not-pep440+local")
print(f"Input: 'not-pep440+local'")
print(f"Type: {type(v2).__name__}")
print(f"LegacyVersion.public: '{v2.public}'")
print(f"Has '+': {'+' in v2.public}")
print()

# Test the specific failing input from the bug report
print("=" * 60)
print("Testing the specific failing input from bug report:")
print("=" * 60)

v3 = pandas.util.version.parse("0.dev0+µ")
print(f"Input: '0.dev0+µ'")
print(f"Type: {type(v3).__name__}")
print(f"public: '{v3.public}'")
print(f"Has '+': {'+' in v3.public}")
print()

# Additional test cases
print("=" * 60)
print("Additional edge cases:")
print("=" * 60)

test_cases = [
    "legacy+version",
    "1.2.3-beta+local",
    "some.version+123",
    "v1+patch"
]

for test_version in test_cases:
    v = pandas.util.version.parse(test_version)
    print(f"Input: '{test_version}'")
    print(f"Type: {type(v).__name__}")
    print(f"public: '{v.public}'")
    print(f"Has '+': {'+' in v.public}")
    print()