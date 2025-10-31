import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.template.backends.jinja2 import get_exception_info


class MockJinjaException:
    def __init__(self, filename, lineno, source, message):
        self.filename = filename
        self.lineno = lineno
        self.source = source
        self.message = message


def test_case(name, source, lineno, expected_content):
    print(f"=== {name} ===")

    # Show original source lines
    print("Original source lines:")
    original_lines = source.split("\n")
    for i, line in enumerate(original_lines, 1):
        marker = " <-- Error here" if i == lineno else ""
        print(f"  Line {i}: {repr(line)}{marker}")

    # Show what happens after strip()
    print("\nAfter strip() in get_exception_info:")
    stripped_source = source.strip()
    stripped_lines = stripped_source.split("\n")
    for i, line in enumerate(stripped_lines, 1):
        print(f"  Line {i}: {repr(line)}")

    print(f"\nJinja2 reports error at line: {lineno}")
    print(f"Number of lines after strip: {len(stripped_lines)}")

    exc = MockJinjaException(
        filename="template.html",
        lineno=lineno,
        source=source,
        message="syntax error"
    )

    try:
        info = get_exception_info(exc)
        print(f"Expected content: {repr(expected_content)}")
        print(f"Actual content: {repr(info['during'])}")
        print(f"Match: {info['during'] == expected_content}")
    except IndexError as e:
        print(f"ERROR: IndexError when accessing line {lineno}: {e}")
        print(f"This happens because line {lineno} doesn't exist after stripping!")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

    print("\n" + "="*50 + "\n")
    return


# Test Case 1: Leading newlines - causes IndexError
test_case(
    "Leading newlines - Line 3 error",
    source="\n\nline 3 content\nline 4 content",
    lineno=3,
    expected_content="line 3 content"
)

# Test Case 2: More severe - Line 4 error with 3 leading newlines
test_case(
    "Three leading newlines - Line 4 error",
    source="\n\n\nline 4 content\nline 5 content",
    lineno=4,
    expected_content="line 4 content"
)

# Test Case 3: Trailing newlines shouldn't cause issues with line access
test_case(
    "Trailing newlines - Line 2 error",
    source="line 1 content\nline 2 content\n\n\n",
    lineno=2,
    expected_content="line 2 content"
)

# Test Case 4: The mismatch issue - wrong line displayed
test_case(
    "Mismatch - Line 3 gets wrong content",
    source="\n\nActual line 3\nActual line 4\nActual line 5",
    lineno=3,
    expected_content="Actual line 3"
)

# Additional test to show the off-by-one nature
print("=== Demonstrating the off-by-N problem ===")
source = "\n\n\nline 4\nline 5\nline 6"
print(f"Source: {repr(source)}")
print("\nOriginal lines (what Jinja2 sees):")
for i, line in enumerate(source.split("\n"), 1):
    print(f"  Line {i}: {repr(line)}")

print("\nAfter strip (what get_exception_info uses):")
for i, line in enumerate(source.strip().split("\n"), 1):
    print(f"  Line {i}: {repr(line)}")

print("\nProblem: If Jinja2 reports error at line 4,")
print("get_exception_info will try to access stripped_lines[3],")
print("but there are only 3 lines after stripping!")