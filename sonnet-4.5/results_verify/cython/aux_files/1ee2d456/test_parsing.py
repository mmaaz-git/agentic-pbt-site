#!/usr/bin/env python3
"""Test to understand how template parsing works"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test various default syntaxes
test_cases = [
    "{{default }}",
    "{{default}}",
    "{{default   }}",
    "{{default\t}}",
    "{{default x}}",
    "{{default x=}}",
    "{{default x=1}}",
]

for content in test_cases:
    print(f"Testing: {repr(content)}")
    try:
        template = Template(content)
        # Try to render to see if it actually works
        try:
            result = template.substitute()
            print(f"  Success - rendered: {repr(result)}")
        except Exception as e:
            print(f"  Template created but render failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  Failed during parsing: {type(e).__name__}: {e}")