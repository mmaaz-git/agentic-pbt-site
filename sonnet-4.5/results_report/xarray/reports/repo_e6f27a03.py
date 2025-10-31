import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import _load_static_files

# Load the CSS content
icons_svg, css_content = _load_static_files()

# Split the CSS into lines
lines = css_content.split('\n')

# Show the problematic line (line 8, 0-indexed as line 7)
print(f"Line 8 of xarray/static/css/style.css:")
print(f"  {lines[7]}")

# Check for the syntax error
if 'var(--pst-color-text-base rgba(' in lines[7]:
    print("\nERROR: Missing comma in CSS var() function!")
    print("  Found: var(--pst-color-text-base rgba(0, 0, 0, 1))")
    print("  Expected: var(--pst-color-text-base, rgba(0, 0, 0, 1))")
    print("\nThis violates CSS Custom Properties specification which requires:")
    print("  var( <custom-property-name> , <declaration-value>? )")
    print("  The comma between property name and fallback value is REQUIRED.")

# Show correct usage in the same file for comparison
print("\nCorrect usage of the same variable elsewhere in the file:")
for i, line in enumerate(lines[:60], 1):
    if 'var(--pst-color-text-base,' in line:
        print(f"  Line {i}: {line.strip()}")