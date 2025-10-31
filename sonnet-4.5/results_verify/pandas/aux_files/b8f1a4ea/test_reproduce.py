from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

print("Testing scientific notation:")
print(f"resolver.size_to_pt('1e-5pt') = {resolver.size_to_pt('1e-5pt')}")
print(f"resolver.size_to_pt('0.00001pt') = {resolver.size_to_pt('0.00001pt')}")

print("\nTesting more examples:")
print(f"resolver.size_to_pt('1e-6pt') = {resolver.size_to_pt('1e-6pt')}")
print(f"resolver.size_to_pt('5e-4pt') = {resolver.size_to_pt('5e-4pt')}")
print(f"resolver.size_to_pt('1.5e-3pt') = {resolver.size_to_pt('1.5e-3pt')}")
print(f"resolver.size_to_pt('1e6pt') = {resolver.size_to_pt('1e6pt')}")