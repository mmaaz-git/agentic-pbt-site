# Bug Report: pandas.core.dtypes.inference.is_re_compilable crashes on invalid regex patterns

**Target**: `pandas.core.dtypes.inference.is_re_compilable`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `is_re_compilable()` function crashes with `re.PatternError` when given invalid regex patterns instead of returning `False` as documented in its API contract.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import re
from pandas.core.dtypes.inference import is_re_compilable


@given(
    pattern=st.one_of(
        st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=10),
        st.integers(),
        st.floats(),
        st.none(),
    )
)
def test_is_re_compilable_consistent_with_re_compile(pattern):
    result = is_re_compilable(pattern)

    try:
        re.compile(pattern)
        can_compile = True
    except (TypeError, re.error):
        can_compile = False

    if can_compile:
        assert result, f"is_re_compilable({pattern!r}) returned False but re.compile succeeded"


if __name__ == "__main__":
    # Run the hypothesis test
    test_is_re_compilable_consistent_with_re_compile()
```

<details>

<summary>
**Failing input**: `'?'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/62
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_is_re_compilable_consistent_with_re_compile FAILED         [100%]

=================================== FAILURES ===================================
_______________ test_is_re_compilable_consistent_with_re_compile _______________

    @given(
>       pattern=st.one_of(
                   ^^^
            st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=10),
            st.integers(),
            st.floats(),
            st.none(),
        )
    )

hypo.py:7:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
hypo.py:15: in test_is_re_compilable_consistent_with_re_compile
    result = is_re_compilable(pattern)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py:188: in is_re_compilable
    re.compile(obj)
/home/npc/miniconda/lib/python3.13/re/__init__.py:289: in compile
    return _compile(pattern, flags)
           ^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/re/__init__.py:350: in _compile
    p = _compiler.compile(pattern, flags)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/re/_compiler.py:748: in compile
    p = _parser.parse(p, flags)
        ^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/re/_parser.py:980: in parse
    p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/npc/miniconda/lib/python3.13/re/_parser.py:459: in _parse_sub
    itemsappend(_parse(source, state, verbose, nested + 1,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

source = <re._parser.Tokenizer object at 0x7fb2bdaedfd0>
state = <re._parser.State object at 0x7fb2bdadcc50>, verbose = 0, nested = 1
first = True

    def _parse(source, state, verbose, nested, first=False):
        # parse a simple pattern
        subpattern = SubPattern(state)

        # precompute constants into local variables
        subpatternappend = subpattern.append
        sourceget = source.get
        sourcematch = source.match
        _len = len
        _ord = ord

        while True:

            this = source.next
            if this is None:
                break # end of pattern
            if this in "|)":
                break # end of subpattern
            sourceget()

            if verbose:
                # skip whitespace and comments
                if this in WHITESPACE:
                    continue
                if this == "#":
                    while True:
                        this = sourceget()
                        if this is None or this == "\n":
                            break
                    continue

            if this[0] == "\\":
                code = _escape(source, this, state)
                subpatternappend(code)

            elif this not in SPECIAL_CHARS:
                subpatternappend((LITERAL, _ord(this)))

            elif this == "[":
                here = source.tell() - 1
                # character set
                set = []
                setappend = set.append
    ##          if sourcematch(":"):
    ##              pass # handle character classes
                if source.next == '[':
                    import warnings
                    warnings.warn(
                        'Possible nested set at position %d' % source.tell(),
                        FutureWarning, stacklevel=nested + 6
                    )
                negate = sourcematch("^")
                # check remaining characters
                while True:
                    this = sourceget()
                    if this is None:
                        raise source.error("unterminated character set",
                                           source.tell() - here)
                    if this == "]" and set:
                        break
                    elif this[0] == "\\":
                        code1 = _class_escape(source, this)
                    else:
                        if set and this in '-&~|' and source.next == this:
                            import warnings
                            warnings.warn(
                                'Possible set %s at position %d' % (
                                    'difference' if this == '-' else
                                    'intersection' if this == '&' else
                                    'symmetric difference' if this == '~' else
                                    'union',
                                    source.tell() - 1),
                                FutureWarning, stacklevel=nested + 6
                            )
                        code1 = LITERAL, _ord(this)
                    if sourcematch("-"):
                        # potential range
                        that = sourceget()
                        if that is None:
                            raise source.error("unterminated character set",
                                               source.tell() - here)
                        if that == "]":
                            if code1[0] is IN:
                                code1 = code1[1][0]
                            setappend(code1)
                            setappend((LITERAL, _ord("-")))
                            break
                        if that[0] == "\\":
                            code2 = _class_escape(source, that)
                        else:
                            if that == '-':
                                import warnings
                                warnings.warn(
                                    'Possible set difference at position %d' % (
                                        source.tell() - 2),
                                    FutureWarning, stacklevel=nested + 6
                                )
                            code2 = LITERAL, _ord(that)
                        if code1[0] != LITERAL or code2[0] != LITERAL:
                            msg = "bad character range %s-%s" % (this, that)
                            raise source.error(msg, len(this) + 1 + len(that))
                        lo = code1[1]
                        hi = code2[1]
                        if hi < lo:
                            msg = "bad character range %s-%s" % (this, that)
                            raise source.error(msg, len(this) + 1 + len(that))
                        setappend((RANGE, (lo, hi)))
                    else:
                        if code1[0] is IN:
                            code1 = code1[1][0]
                        setappend(code1)

                set = _uniq(set)
                # XXX: <fl> should move set optimization to compiler!
                if _len(set) == 1 and set[0][0] is LITERAL:
                    # optimization
                    if negate:
                        subpatternappend((NOT_LITERAL, set[0][1]))
                    else:
                        subpatternappend(set[0])
                else:
                    if negate:
                        set.insert(0, (NEGATE, None))
                    # charmap optimization can't be added here because
                    # global flags still are not known
                    subpatternappend((IN, set))

            elif this in REPEAT_CHARS:
                # repeat previous item
                here = source.tell()
                if this == "?":
                    min, max = 0, 1
                elif this == "*":
                    min, max = 0, MAXREPEAT

                elif this == "+":
                    min, max = 1, MAXREPEAT
                elif this == "{":
                    if source.next == "}":
                        subpatternappend((LITERAL, _ord(this)))
                        continue

                    min, max = 0, MAXREPEAT
                    lo = hi = ""
                    while source.next in DIGITS:
                        lo += sourceget()
                    if sourcematch(","):
                        while source.next in DIGITS:
                            hi += sourceget()
                    else:
                        hi = lo
                    if not sourcematch("}"):
                        subpatternappend((LITERAL, _ord(this)))
                        source.seek(here)
                        continue

                    if lo:
                        min = int(lo)
                        if min >= MAXREPEAT:
                            raise OverflowError("the repetition number is too large")
                    if hi:
                        max = int(hi)
                        if max >= MAXREPEAT:
                            raise OverflowError("the repetition number is too large")
                        if max < min:
                            raise source.error("min repeat greater than max repeat",
                                               source.tell() - here)
                else:
                    raise AssertionError("unsupported quantifier %r" % (char,))
                # figure out which item to repeat
                if subpattern:
                    item = subpattern[-1:]
                else:
                    item = None
                if not item or item[0][0] is AT:
>                   raise source.error("nothing to repeat",
                                       source.tell() - here + len(this))
E                   re.PatternError: nothing to repeat at position 0
E                   Falsifying example: test_is_re_compilable_consistent_with_re_compile(
E                       pattern='?',
E                   )
E                   Explanation:
E                       These lines were always and only run by failing examples:
E                           /home/npc/miniconda/lib/python3.13/re/_constants.py:38
E                           /home/npc/miniconda/lib/python3.13/re/_parser.py:549
E                           /home/npc/miniconda/lib/python3.13/re/_parser.py:641
E                           /home/npc/miniconda/lib/python3.13/re/_parser.py:685
E                           /home/npc/miniconda/lib/python3.13/re/_parser.py:686

/home/npc/miniconda/lib/python3.13/re/_parser.py:686: PatternError
=========================== short test summary info ============================
FAILED hypo.py::test_is_re_compilable_consistent_with_re_compile - re.Pattern...
============================== 1 failed in 0.94s ===============================
```
</details>

## Reproducing the Bug

```python
from pandas.core.dtypes.inference import is_re_compilable

# Test with invalid regex pattern that should return False but crashes
result = is_re_compilable("(")
print(f"Result: {result}")
```

<details>

<summary>
Crashes with re.PatternError: missing ), unterminated subpattern
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/62/repo.py", line 4, in <module>
    result = is_re_compilable("(")
  File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/core/dtypes/inference.py", line 188, in is_re_compilable
    re.compile(obj)
    ~~~~~~~~~~^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 289, in compile
    return _compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/__init__.py", line 350, in _compile
    p = _compiler.compile(pattern, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_compiler.py", line 748, in compile
    p = _parser.parse(p, flags)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 980, in parse
    p = _parse_sub(source, state, flags & SRE_FLAG_VERBOSE, 0)
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 459, in _parse_sub
    itemsappend(_parse(source, state, verbose, nested + 1,
                ~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       not nested and not items))
                       ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/re/_parser.py", line 865, in _parse
    raise source.error("missing ), unterminated subpattern",
                       source.tell() - start)
re.PatternError: missing ), unterminated subpattern at position 0
```
</details>

## Why This Is A Bug

The function `is_re_compilable` violates its documented contract and the semantic expectations of a boolean predicate function. According to its docstring, it should "Check if the object can be compiled into a regex pattern instance" and return a boolean value indicating "Whether `obj` can be compiled as a regex pattern."

The function name follows the standard `is_*` convention for boolean predicates, which by convention should not raise exceptions but return `True` or `False`. Currently, the function only catches `TypeError` exceptions (for non-string inputs) but fails to catch `re.error` exceptions that occur when `re.compile()` encounters invalid regex syntax. This causes the function to crash instead of returning `False` for patterns like `"("`, `")"`, `"?"`, `"*"`, and `"["` - all of which are strings that cannot be compiled as valid regex patterns.

The entire purpose of this function is to provide a safe way to check if something can be compiled as a regex without risking an exception. Users calling a function named `is_re_compilable` reasonably expect it to return `False` for invalid patterns, not crash with an unhandled exception.

## Relevant Context

This function is part of the public pandas API (`pandas.api.types.is_re_compilable`) and is used for type checking and validation. The bug affects any code that uses this function to validate user input or check whether strings are valid regex patterns before attempting to use them.

Common invalid regex patterns that trigger this bug include:
- Unmatched parentheses: `"("`, `")"`
- Quantifiers without preceding elements: `"?"`, `"*"`, `"+"`
- Unmatched brackets: `"["`, `"]"`
- Invalid escape sequences and other malformed regex syntax

The fix is straightforward - the exception handler needs to catch both `TypeError` (already handled) and `re.error` (the base class for regex compilation errors, which includes `re.PatternError` in Python 3.11+).

## Proposed Fix

```diff
--- a/pandas/core/dtypes/inference.py
+++ b/pandas/core/dtypes/inference.py
@@ -186,7 +186,7 @@ def is_re_compilable(obj) -> bool:
     """
     try:
         re.compile(obj)
-    except TypeError:
+    except (TypeError, re.error):
         return False
     else:
         return True
```