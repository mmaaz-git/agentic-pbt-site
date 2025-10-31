import re

regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"
gdb_output = "GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 7.2"

match = re.match(regex, gdb_output)
version = list(map(int, match.groups()))

print(f"Detected version: {version}")
print(f"Expected: [7, 2]")
print(f"Bug: Matched Ubuntu version [12, 1] instead of GDB version [7, 2]")

if version >= [7, 2]:
    print("✓ Would enable tests (correct in this case, but only by coincidence)")
else:
    print("✗ Would skip tests despite having GDB 7.2")