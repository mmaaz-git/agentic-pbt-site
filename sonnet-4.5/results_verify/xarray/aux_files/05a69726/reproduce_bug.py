import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core.formatting_html import collapsible_section

result_enabled = collapsible_section("Coords", "", "", n_items=5, enabled=True)
result_disabled = collapsible_section("Dims", "", "", n_items=0, enabled=True)

print("Enabled checkbox (n_items=5):")
print(f"  Has tooltip: {'title=' in result_enabled}")
print(f"  Has 'disabled': {'disabled' in result_enabled}")

print("\nDisabled checkbox (n_items=0):")
print(f"  Has tooltip: {'title=' in result_disabled}")
print(f"  Has 'disabled': {'disabled' in result_disabled}")

# Let's also check the actual HTML output
print("\n--- Actual HTML output ---")
print("Enabled checkbox HTML snippet:")
print(result_enabled[:200] + "...")
print("\nDisabled checkbox HTML snippet:")
print(result_disabled[:200] + "...")