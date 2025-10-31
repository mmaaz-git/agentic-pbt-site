#!/usr/bin/env python3
"""Test to understand strip_string_literals behavior"""

import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages")

from Cython.Build.Dependencies import strip_string_literals

test_cases = [
    "foo bar",
    "foo # bar",
    "# foo",
    "#",
    "foo 'bar' baz",
    'foo "bar" baz',
    "foo # 'bar'",
    "'foo' # bar",
]

for test in test_cases:
    result, literals = strip_string_literals(test)
    print(f"Input: '{test}'")
    print(f"Result: '{result}'")
    print(f"Literals: {literals}")
    print()