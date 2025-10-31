#!/usr/bin/env python3
"""Test the reported bug in old_build_ext option precedence"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from distutils.dist import Distribution
from Cython.Distutils import Extension
from Cython.Distutils.old_build_ext import old_build_ext

def test_reproduction():
    """Reproduce the bug as reported"""
    print("=== Reproducing the Bug ===")

    dist = Distribution()
    cmd = old_build_ext(dist)
    cmd.initialize_options()
    cmd.finalize_options()

    # Set command-line value to 0 (falsy)
    cmd.cython_create_listing = 0

    # Create extension with attribute set to True
    ext = Extension("test", ["test.pyx"], cython_create_listing=True)

    # Simulate how cython_sources() merges the options (line 223-224)
    create_listing = cmd.cython_create_listing or getattr(ext, 'cython_create_listing', 0)

    print(f"Command value: {cmd.cython_create_listing}")
    print(f"Extension value: {ext.cython_create_listing}")
    print(f"Result with 'or': {create_listing}")
    print(f"Expected result: {cmd.cython_create_listing} (command value should take precedence)")

    if create_listing != cmd.cython_create_listing:
        print("\nBUG CONFIRMED: Command value was overridden by extension value!")
    else:
        print("\nNo bug: Command value took precedence as expected")

    return create_listing

def test_hypothesis():
    """Run the hypothesis-style test from the report"""
    print("\n=== Running Hypothesis-style Test ===")

    def test_create_listing_or_operator_bug(ext_value):
        """Command value should take precedence over extension value, even when falsy."""
        dist = Distribution()
        cmd = old_build_ext(dist)
        cmd.initialize_options()
        cmd.finalize_options()

        cmd.cython_create_listing = 0
        ext = Extension("test", ["test.pyx"], cython_create_listing=ext_value)

        create_listing = cmd.cython_create_listing or getattr(ext, 'cython_create_listing', 0)

        assert create_listing == 0, f"Bug: expected 0 (cmd value), got {create_listing}"

    # Test with ext_value=True (the failing case)
    try:
        test_create_listing_or_operator_bug(True)
        print("Test passed with ext_value=True")
    except AssertionError as e:
        print(f"Test failed with ext_value=True: {e}")

    # Test with ext_value=False
    try:
        test_create_listing_or_operator_bug(False)
        print("Test passed with ext_value=False")
    except AssertionError as e:
        print(f"Test failed with ext_value=False: {e}")

def test_all_affected_options():
    """Test all options that use the same pattern"""
    print("\n=== Testing All Affected Options ===")

    dist = Distribution()
    cmd = old_build_ext(dist)
    cmd.initialize_options()
    cmd.finalize_options()

    # Set all command options to falsy values
    cmd.cython_create_listing = 0
    cmd.cython_line_directives = 0
    cmd.no_c_in_traceback = 0
    cmd.cython_cplus = 0
    cmd.cython_gen_pxi = 0
    cmd.cython_gdb = False
    cmd.cython_compile_time_env = None

    # Create extension with all options set to truthy values
    ext = Extension("test", ["test.pyx"],
                   cython_create_listing=1,
                   cython_line_directives=1,
                   no_c_in_traceback=1,
                   cython_cplus=1,
                   cython_gen_pxi=1,
                   cython_gdb=True,
                   cython_compile_time_env={'TEST': 1})

    # Test each option using the same pattern from cython_sources()
    issues = []

    # Line 223-224
    create_listing = cmd.cython_create_listing or getattr(ext, 'cython_create_listing', 0)
    if create_listing != cmd.cython_create_listing:
        issues.append(f"cython_create_listing: cmd={cmd.cython_create_listing}, ext={ext.cython_create_listing}, result={create_listing}")

    # Line 225-226
    line_directives = cmd.cython_line_directives or getattr(ext, 'cython_line_directives', 0)
    if line_directives != cmd.cython_line_directives:
        issues.append(f"cython_line_directives: cmd={cmd.cython_line_directives}, ext={ext.cython_line_directives}, result={line_directives}")

    # Line 227-228
    no_c_in_traceback = cmd.no_c_in_traceback or getattr(ext, 'no_c_in_traceback', 0)
    if no_c_in_traceback != cmd.no_c_in_traceback:
        issues.append(f"no_c_in_traceback: cmd={cmd.no_c_in_traceback}, ext={ext.no_c_in_traceback}, result={no_c_in_traceback}")

    # Line 229-230 (special case with language check)
    cplus = cmd.cython_cplus or getattr(ext, 'cython_cplus', 0) or (ext.language and ext.language.lower() == 'c++')
    if cplus != cmd.cython_cplus and not (ext.language and ext.language.lower() == 'c++'):
        issues.append(f"cython_cplus: cmd={cmd.cython_cplus}, ext={ext.cython_cplus}, result={cplus}")

    # Line 231
    cython_gen_pxi = cmd.cython_gen_pxi or getattr(ext, 'cython_gen_pxi', 0)
    if cython_gen_pxi != cmd.cython_gen_pxi:
        issues.append(f"cython_gen_pxi: cmd={cmd.cython_gen_pxi}, ext={ext.cython_gen_pxi}, result={cython_gen_pxi}")

    # Line 232
    cython_gdb = cmd.cython_gdb or getattr(ext, 'cython_gdb', False)
    if cython_gdb != cmd.cython_gdb:
        issues.append(f"cython_gdb: cmd={cmd.cython_gdb}, ext={ext.cython_gdb}, result={cython_gdb}")

    # Line 233-234
    cython_compile_time_env = cmd.cython_compile_time_env or getattr(ext, 'cython_compile_time_env', None)
    if cython_compile_time_env != cmd.cython_compile_time_env:
        issues.append(f"cython_compile_time_env: cmd={cmd.cython_compile_time_env}, ext={ext.cython_compile_time_env}, result={cython_compile_time_env}")

    if issues:
        print("Issues found (command values overridden by extension values):")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues found")

    return len(issues) > 0

if __name__ == "__main__":
    test_reproduction()
    test_hypothesis()
    test_all_affected_options()