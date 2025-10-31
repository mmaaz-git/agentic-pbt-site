from django.utils.translation import to_locale

# Test case demonstrating the bug with uppercase input without dash
print(f"to_locale('AAAA') = {to_locale('AAAA')!r}")
print(f"to_locale('aaaa') = {to_locale('aaaa')!r}")

# Show that they produce different results
assert to_locale('AAAA') == to_locale('aaaa'), "Case should not matter"