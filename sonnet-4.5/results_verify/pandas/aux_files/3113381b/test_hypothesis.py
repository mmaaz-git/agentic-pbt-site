#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
import pandas.util.version as version_module

@st.composite
def valid_version_strings(draw):
    ep = draw(st.one_of(st.just(None), st.integers(min_value=0, max_value=10)))
    rel = draw(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=5))
    has_pre = draw(st.booleans())
    has_post = draw(st.booleans())
    has_dev = draw(st.booleans())
    has_local = draw(st.booleans())

    parts = []
    if ep and ep != 0:
        parts.append(f"{ep}!")
    parts.append(".".join(str(x) for x in rel))

    if has_pre:
        pre_type = draw(st.sampled_from(["a", "b", "rc"]))
        pre_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f"{pre_type}{pre_num}")

    if has_post:
        post_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f".post{post_num}")

    if has_dev:
        dev_num = draw(st.integers(min_value=0, max_value=100))
        parts.append(f".dev{dev_num}")

    if has_local:
        local = draw(st.lists(st.text(alphabet=st.characters(whitelist_categories=('Ll', 'Lu', 'Nd')), min_size=1, max_size=10), min_size=1, max_size=3))
        parts.append("+" + ".".join(local))

    return "".join(parts)

# Run test with the specific failing example
print("Testing specific failing input: '0.dev0+µ'")
try:
    v = version_module.parse('0.dev0+µ')
    public = v.public
    print(f"Version type: {type(v).__name__}")
    print(f"Public version: '{public}'")
    print(f"Contains '+': {'+' in public}")
    assert "+" not in public, f"public version contains local part: {public}"
    print("✓ Test passed")
except AssertionError as e:
    print(f"✗ Test failed: {e}")
print()

# Run a few more examples
@given(valid_version_strings())
@settings(max_examples=10)
@example('0.dev0+µ')  # Include the known failing case
def test_public_version_no_local(version_str):
    v = version_module.parse(version_str)
    public = v.public
    assert "+" not in public, f"public version contains local part: {public} (from input: {version_str}, type: {type(v).__name__})"

print("Running hypothesis tests...")
try:
    test_public_version_no_local()
    print("All tests passed!")
except AssertionError as e:
    print(f"Tests failed: {e}")