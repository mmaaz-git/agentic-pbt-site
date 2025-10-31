import sys
import re

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files


def test_css_var_function_syntax():
    icons_svg, css_content = _load_static_files()

    lines = css_content.split('\n')
    invalid_lines = []

    for line_num, line in enumerate(lines, 1):
        if 'var(' in line and 'rgba(' in line:
            # Look for var() calls that have rgba() as fallback but missing comma
            var_match = re.search(r'var\(--[a-zA-Z0-9-]+\s+rgba\(', line)
            if var_match:
                invalid_lines.append((line_num, line.strip()))

    assert not invalid_lines, f"Found CSS var() calls with missing comma before fallback value:\n" + \
        "\n".join([f"  Line {num}: {line}" for num, line in invalid_lines])


if __name__ == "__main__":
    try:
        test_css_var_function_syntax()
        print("Test passed: No CSS syntax errors found")
    except AssertionError as e:
        print(f"Test failed: {e}")