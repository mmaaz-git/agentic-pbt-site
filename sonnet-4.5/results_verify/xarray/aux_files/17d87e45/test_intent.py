"""
Test to understand the intended behavior of tooltips in collapsible sections.

From a UX perspective:
1. Tooltips should help users understand what they can do with an element
2. A disabled checkbox cannot be interacted with - no tooltip is needed
3. An enabled checkbox can be clicked to expand/collapse - a tooltip would be helpful

This seems like a genuine bug where the tooltip logic is inverted.
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

# Test tooltip behavior makes UX sense
print("Testing UX logic:")
print("-" * 50)

# Case 1: Enabled checkbox that can be clicked
result = collapsible_section("Test", "", "", n_items=5, enabled=True)
print("Enabled checkbox (can be clicked):")
print(f"  HTML contains 'disabled': {'disabled' in result}")
print(f"  HTML contains tooltip: {'title=' in result}")
print(f"  UX expectation: Should have tooltip to help user understand it's clickable")
print()

# Case 2: Disabled checkbox that cannot be clicked
result = collapsible_section("Test", "", "", n_items=0, enabled=True)
print("Disabled checkbox (cannot be clicked):")
print(f"  HTML contains 'disabled': {'disabled' in result}")
print(f"  HTML contains tooltip: {'title=' in result}")
print(f"  UX expectation: Should NOT have tooltip since it cannot be interacted with")
print()

# Let's also test how this function is called in real usage
print("Real usage in dim_section:")
print("-" * 50)
# In dim_section (line 230), the function is called with enabled=False
result = collapsible_section("Dimensions", inline_details="(x: 10, y: 20)", enabled=False, collapsed=True)
print(f"  HTML contains 'disabled': {'disabled' in result}")
print(f"  HTML contains tooltip: {'title=' in result}")
print("  This creates a permanently disabled section with a misleading tooltip")