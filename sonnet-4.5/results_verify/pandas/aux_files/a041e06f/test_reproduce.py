import pandas.api.types as pat

print("Testing valid regex:")
print(f"pat.is_re_compilable('.*') = {pat.is_re_compilable('.*')}")

print("\nTesting invalid regex (unmatched parenthesis):")
try:
    result = pat.is_re_compilable('(')
    print(f"pat.is_re_compilable('(') = {result}")
except Exception as e:
    print(f"pat.is_re_compilable('(') raised {type(e).__name__}: {e}")