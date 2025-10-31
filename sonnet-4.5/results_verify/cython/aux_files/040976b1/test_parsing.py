#!/usr/bin/env python3
"""Test to see how the parser handles multiple else clauses"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template
from Cython.Tempita._tempita import parse

# Test parsing multiple else clauses
content = """
{{if x}}
true_branch
{{else}}
first_else
{{else}}
second_else
{{endif}}
"""

template = Template(content)

# Check the parsed structure
print("Template source:")
print(content)

print("\nParsed structure:")
try:
    # Access the internal parsed structure
    print("Template._parsed:", template._parsed)
except AttributeError:
    print("Could not access _parsed attribute")

# Try to parse directly
print("\nDirect parsing test:")
tokens = template._parse(content, name='test')
print("Parsed tokens:", tokens)