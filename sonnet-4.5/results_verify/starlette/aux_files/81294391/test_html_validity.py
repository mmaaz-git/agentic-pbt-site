"""Test if browsers and parsers handle &nbsp without semicolon"""
import html.parser

# Test 1: How does Python's html parser handle it?
class TestParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.data = []

    def handle_data(self, data):
        self.data.append(data)

# Test with semicolon (correct)
parser1 = TestParser()
parser1.feed("<p>hello&nbsp;world</p>")
print("With semicolon:", parser1.data)

# Test without semicolon (incorrect)
parser2 = TestParser()
parser2.feed("<p>hello&nbspworld</p>")
print("Without semicolon:", parser2.data)

# Test html.unescape behavior
print("\nhtml.unescape tests:")
print("With semicolon:", repr(html.unescape("hello&nbsp;world")))
print("Without semicolon:", repr(html.unescape("hello&nbspworld")))