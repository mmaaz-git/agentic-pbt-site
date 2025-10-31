import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.template.backends.jinja2 import get_exception_info


class MockJinjaException:
    def __init__(self, filename, lineno, source, message):
        self.filename = filename
        self.lineno = lineno
        self.source = source
        self.message = message


# Test Case 1: Leading newlines
print("=== Test Case 1: Leading newlines ===")
source_with_leading_newlines = "\n\nline 3 content\nline 4 content"

exc = MockJinjaException(
    filename="template.html",
    lineno=3,
    source=source_with_leading_newlines,
    message="syntax error"
)

info = get_exception_info(exc)

print(f"Source: {repr(source_with_leading_newlines)}")
print(f"Jinja2 reported line: {exc.lineno}")
print(f"Expected content: 'line 3 content'")
print(f"Actual content: {repr(info['during'])}")
print(f"Match: {info['during'] == 'line 3 content'}")
print()

# Test Case 2: Trailing newlines
print("=== Test Case 2: Trailing newlines ===")
source_with_trailing_newlines = "line 1 content\nline 2 content\n\n\n"

exc2 = MockJinjaException(
    filename="template.html",
    lineno=2,
    source=source_with_trailing_newlines,
    message="syntax error"
)

info2 = get_exception_info(exc2)

print(f"Source: {repr(source_with_trailing_newlines)}")
print(f"Jinja2 reported line: {exc2.lineno}")
print(f"Expected content: 'line 2 content'")
print(f"Actual content: {repr(info2['during'])}")
print(f"Match: {info2['during'] == 'line 2 content'}")
print()

# Test Case 3: Both leading and trailing whitespace
print("=== Test Case 3: Both leading and trailing whitespace ===")
source_with_both = "\n\n\nline 4 content\nline 5 content\n\n"

exc3 = MockJinjaException(
    filename="template.html",
    lineno=4,
    source=source_with_both,
    message="syntax error"
)

info3 = get_exception_info(exc3)

print(f"Source: {repr(source_with_both)}")
print(f"Jinja2 reported line: {exc3.lineno}")
print(f"Expected content: 'line 4 content'")
print(f"Actual content: {repr(info3['during'])}")
print(f"Match: {info3['during'] == 'line 4 content'}")
print()

# Let's also verify what lines are actually in the source
print("=== Verification: Line by line analysis ===")
print("Source with leading newlines - actual lines:")
lines = source_with_leading_newlines.split("\n")
for i, line in enumerate(lines, 1):
    print(f"  Line {i}: {repr(line)}")
print()

print("After strip() - what get_exception_info sees:")
stripped_lines = source_with_leading_newlines.strip().split("\n")
for i, line in enumerate(stripped_lines, 1):
    print(f"  Line {i}: {repr(line)}")