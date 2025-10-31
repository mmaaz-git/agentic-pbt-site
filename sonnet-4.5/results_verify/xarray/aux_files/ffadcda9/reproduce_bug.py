from xarray.core.formatting_html import collapsible_section

result = collapsible_section("Test Section", n_items=-1)

print("Result HTML snippet:")
print(result[:500])
print("\n" + "="*50 + "\n")
print("Contains 'disabled':", "disabled" in result)
print("Contains 'checked':", "checked" in result)
print("Displays count:", "(-1)" in result)