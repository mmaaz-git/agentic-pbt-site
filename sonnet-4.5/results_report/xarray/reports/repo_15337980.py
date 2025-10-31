import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

# Test with enabled checkbox (should have tooltip but doesn't)
result_enabled = collapsible_section("Coords", "", "", n_items=5, enabled=True)
print("Enabled checkbox (n_items=5, enabled=True):")
print(f"  HTML output contains tooltip: {'title=' in result_enabled}")
print(f"  HTML output contains 'disabled': {'disabled' in result_enabled}")
print()

# Test with disabled checkbox (shouldn't have tooltip but does)
result_disabled = collapsible_section("Dims", "", "", n_items=0, enabled=True)
print("Disabled checkbox (n_items=0, enabled=True):")
print(f"  HTML output contains tooltip: {'title=' in result_disabled}")
print(f"  HTML output contains 'disabled': {'disabled' in result_disabled}")
print()

# Show the actual HTML for clarity
print("Actual HTML for enabled checkbox:")
print(result_enabled)
print()
print("Actual HTML for disabled checkbox:")
print(result_disabled)