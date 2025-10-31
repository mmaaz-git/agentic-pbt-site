import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

icons_svg, css_content = _load_static_files()
lines = css_content.split('\n')

print("Line 8:", lines[7])
print()
print("Looking for other uses of the same variable pattern:")
for i, line in enumerate(lines, 1):
    if '--pst-color-text-base' in line:
        print(f"Line {i}: {line.strip()}")