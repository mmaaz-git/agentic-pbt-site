test_str = "line1\nline2\n\nline3"
lines = test_str.split('\n')
print("Split result:")
for i, line in enumerate(lines):
    print(f"  [{i}]: '{line}'")