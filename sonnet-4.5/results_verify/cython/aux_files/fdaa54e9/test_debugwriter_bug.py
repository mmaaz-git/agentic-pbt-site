#!/usr/bin/env python3

import tempfile
import shutil
from hypothesis import given, strategies as st, settings
import sys
import os
import traceback

# Add cython to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

from Cython.Debugger.DebugWriter import CythonDebugWriter

valid_xml_start = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
valid_xml_chars = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
valid_xml_tag = st.builds(
    lambda start, rest: start + rest,
    valid_xml_start,
    st.text(alphabet=valid_xml_chars, max_size=20)
)

@given(valid_xml_tag, st.dictionaries(valid_xml_tag, st.text()))
@settings(max_examples=100)
def test_add_entry_with_attrs(tag_name, attrs):
    tmpdir = tempfile.mkdtemp()
    try:
        writer = CythonDebugWriter(tmpdir)
        writer.tb.start('Module', {})
        writer.add_entry(tag_name, **attrs)
        writer.tb.end('Module')
        writer.tb.end('cython_debug')
        writer.tb.close()
    except Exception as e:
        print(f"Exception with tag_name='{tag_name}', attrs={repr(attrs)}: {e}")
        if '\x1f' in str(attrs.values()):
            print("Found control character in attributes")
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# Run hypothesis test
print("Running hypothesis test...")
try:
    test_add_entry_with_attrs()
    print("Hypothesis test passed without finding issues")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Test specific failing input from the bug report
print("\nTesting specific failing input: tag_name='a', attrs={'a': '\\x1f'}")
tmpdir = tempfile.mkdtemp()
try:
    writer = CythonDebugWriter(tmpdir)
    writer.tb.start('Module', {})
    writer.add_entry('a', a='\x1f')
    writer.tb.end('Module')
    writer.tb.end('cython_debug')
    writer.tb.close()
    print("Test passed - no crash")
except Exception as e:
    print(f"Test crashed with: {e}")
    traceback.print_exc()
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)