import html

# Demonstrating the bug in Starlette's ServerErrorMiddleware
# This shows how spaces are replaced with invalid HTML entities

def demonstrate_bug():
    # Example line of code that might appear in a traceback
    line = "    def example_function():"

    # This is what Starlette currently does (line 191 in errors.py)
    buggy_result = html.escape(line).replace(" ", "&nbsp")

    # This is what it should do
    correct_result = html.escape(line).replace(" ", "&nbsp;")

    print("Original line:")
    print(f"'{line}'")
    print()

    print("Buggy result (missing semicolons):")
    print(f"'{buggy_result}'")
    print()

    print("Correct result (with semicolons):")
    print(f"'{correct_result}'")
    print()

    # Demonstrate the issue with another example
    print("=" * 50)
    print("Another example with multiple spaces:")
    line2 = "        return x + y  # Add two numbers"
    buggy_result2 = html.escape(line2).replace(" ", "&nbsp")
    correct_result2 = html.escape(line2).replace(" ", "&nbsp;")

    print(f"Original: '{line2}'")
    print(f"Buggy:    '{buggy_result2}'")
    print(f"Correct:  '{correct_result2}'")
    print()

    # Show the difference
    print("The difference:")
    print(f"Invalid entity: '&nbsp' (no semicolon)")
    print(f"Valid entity:   '&nbsp;' (with semicolon)")
    print()
    print("According to HTML spec, named character references MUST end with semicolon.")

if __name__ == "__main__":
    demonstrate_bug()