#!/usr/bin/env python3
"""Verify HTML5 standard compliance for character entities."""

import html.parser

def test_html_parser_behavior():
    """Test how Python's HTML parser handles incomplete entities."""

    test_cases = [
        ("&nbsp;", "Complete entity with semicolon"),
        ("&nbsp", "Incomplete entity without semicolon"),
        ("&nbsp&nbsp", "Multiple incomplete entities"),
        ("&nbsp;&nbsp;", "Multiple complete entities"),
        ("test &nbsp test", "Incomplete entity in text"),
        ("test &nbsp; test", "Complete entity in text"),
    ]

    print("Testing HTML parser behavior with different entity formats:")
    print("=" * 60)

    for test_html, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Input: {repr(test_html)}")

        parser = html.parser.HTMLParser()

        # Track what gets parsed
        parsed_data = []
        original_handle_data = parser.handle_data

        def capture_data(data):
            parsed_data.append(data)
            original_handle_data(data)

        parser.handle_data = capture_data

        try:
            parser.feed(f"<p>{test_html}</p>")
            print(f"Parsed successfully")
            if parsed_data:
                print(f"Parsed data: {repr(''.join(parsed_data))}")
        except Exception as e:
            print(f"Parser error: {e}")

    print("\n" + "=" * 60)
    print("HTML5 Standard Requirements:")
    print("- Named character references MUST end with a semicolon")
    print("- While browsers may parse &nbsp without semicolon for backwards compatibility,")
    print("  this is considered a parse error in HTML5")
    print("- Strict validators will flag this as invalid HTML")


def test_browser_rendering():
    """Generate test HTML to check browser rendering."""

    html_with_bug = """<!DOCTYPE html>
<html>
<head><title>Entity Test</title></head>
<body>
<h2>Incomplete entities (INVALID HTML5):</h2>
<p>Four spaces: '&nbsp&nbsp&nbsp&nbsp'</p>
<p>Text with spaces: 'Hello&nbspWorld'</p>

<h2>Complete entities (VALID HTML5):</h2>
<p>Four spaces: '&nbsp;&nbsp;&nbsp;&nbsp;'</p>
<p>Text with spaces: 'Hello&nbsp;World'</p>
</body>
</html>"""

    with open('/home/npc/pbt/agentic-pbt/worker_/31/test_entities.html', 'w') as f:
        f.write(html_with_bug)

    print("\nGenerated test HTML file: test_entities.html")
    print("This demonstrates the difference between complete and incomplete entities.")
    print("\nNote: While modern browsers may render both correctly due to error recovery,")
    print("the incomplete entities violate HTML5 standards and may cause issues with:")
    print("- HTML validators")
    print("- Accessibility tools")
    print("- Strict parsers")
    print("- Future browser versions")


if __name__ == "__main__":
    test_html_parser_behavior()
    test_browser_rendering()