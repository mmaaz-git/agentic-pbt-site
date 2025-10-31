#!/usr/bin/env python3
"""Test Django code from virtualenv for typo"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Read the actual Django file directly
with open('/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/backends/oracle/operations.py', 'r') as f:
    content = f.read()

# Search for the typo
if 'Invalid loookup type' in content:
    print("✓ Typo confirmed in Django source: Found 'Invalid loookup type' with 3 o's")
    # Find the line number
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'Invalid loookup type' in line:
            print(f"  Found on line {i}: {line.strip()}")
else:
    print("✗ Typo not found in Django source")

# Also check if correct spelling exists anywhere
if 'Invalid lookup type' in content:
    print("Note: Correct spelling 'Invalid lookup type' also found in the file")