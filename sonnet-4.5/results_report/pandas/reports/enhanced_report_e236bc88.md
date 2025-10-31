# Bug Report: pandas.errors.AbstractMethodError Swapped Variables in Error Message

**Target**: `pandas.errors.AbstractMethodError`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When passing an invalid `methodtype` parameter to `AbstractMethodError.__init__`, the error message has swapped variables, displaying the invalid value where valid options should be listed and vice versa.

## Property-Based Test

```python
from hypothesis import given, strategies as st, example
import pandas.errors
import pytest


@given(st.text().filter(lambda x: x not in ['method', 'classmethod', 'staticmethod', 'property']))
@example('invalid_type')  # Specific example to demonstrate the bug
def test_abstract_method_error_invalid_types(invalid_methodtype):
    """Test that AbstractMethodError produces correct error message for invalid methodtypes."""
    class DummyClass:
        pass

    instance = DummyClass()

    with pytest.raises(ValueError) as exc_info:
        pandas.errors.AbstractMethodError(instance, methodtype=invalid_methodtype)

    error_message = str(exc_info.value)

    # The error message should be:
    # "methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}, got invalid_methodtype instead."
    # But currently it is:
    # "methodtype must be one of invalid_methodtype, got {'method', 'classmethod', 'staticmethod', 'property'} instead."

    # Check that the message contains the required parts
    assert 'methodtype must be one of' in error_message

    # This assertion would fail with the current bug - the message has variables swapped
    # We expect: "methodtype must be one of {valid_types}, got {invalid_methodtype} instead."
    # But we get: "methodtype must be one of {invalid_methodtype}, got {valid_types} instead."

    # For now, just document what we actually get
    print(f"Input: {repr(invalid_methodtype)}")
    print(f"Error message: {error_message}")


if __name__ == '__main__':
    # Run with a specific example
    test_abstract_method_error_invalid_types('invalid_type')
```

<details>

<summary>
**Failing input**: `invalid_methodtype='invalid_type'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/18
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_abstract_method_error_invalid_types Input: 'invalid_type'
Error message: methodtype must be one of invalid_type, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: ''
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x89ù8\U000c4d477;'
Error message: methodtype must be one of ù8󄵇7;, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Lï\x92¬\U0008088a'
Error message: methodtype must be one of Lï¬򀢊, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'x\U000acf57'
Error message: methodtype must be one of x򬽗, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'j\U000c70c6'
Error message: methodtype must be one of j󇃆, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000c2c44\x03\x0c2\U000732dd\U00060300Ó\x849'
Error message: methodtype must be one of 󂱄2񳋝񠌀Ó9, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Ú'
Error message: methodtype must be one of Ú, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x9b\x96'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\\/+'
Error message: methodtype must be one of \/+, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '𢚳e𫭑\U0010948a\x04\U000b832cf'
Error message: methodtype must be one of 𢚳e𫭑􉒊򸌬f, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x84\U000aa745\U0003ebda\x85¯\U0003e8e2Ð|9&'
Error message: methodtype must be one of 򪝅𾯚¯𾣢Ð|9&, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'P'
Error message: methodtype must be one of P, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0010b0587$Ë}'
Error message: methodtype must be one of 􋁘7$Ë}, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x1b\U000d39f2\x9bÞ¶\U000192dc\U00077356\x16\x1dï\x02O\x1a{?ÒûD¾\x03\x8em'
Error message: methodtype must be one of 󓧲Þ¶𙋜񷍖ïO{?ÒûD¾m, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x98'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Á\x01Õs\x92'
Error message: methodtype must be one of ÁÕs, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ì'
Error message: methodtype must be one of ì, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x91\U0008fc1a\U000e1972\U000e10aen7Ê[\x85õA'
Error message: methodtype must be one of 򏰚󡥲󡂮n7Ê[õA, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000c83f6\U000447de\U00083ed3\x93g'
Error message: methodtype must be one of 󈏶񄟞򃻓g, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000f42cb®\x8b\x0c\x87'
Error message: methodtype must be one of 󴋋®, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0001a287'
Error message: methodtype must be one of 𚊇, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '1áf\U000b2710n𨚖Ô'
Error message: methodtype must be one of 1áf򲜐n𨚖Ô, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x99y'
Error message: methodtype must be one of y, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'è¶\xadù'
Error message: methodtype must be one of è¶­ù, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '÷Í'
Error message: methodtype must be one of ÷Í, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\n '
Error message: methodtype must be one of
 , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000914a8\x04ê\U001095e4'
Error message: methodtype must be one of 򑒨ê􉗤, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Ê]|Ô\U0007bd83\U000d85225y'
Error message: methodtype must be one of Ê]|Ô񻶃󘔢5y, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000a0a11£.\U0007087fa:k¡\x98\x05#\x8a'
Error message: methodtype must be one of 򠨑£.񰡿a:k¡#, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '훺\U000113e4¬ï\x06|'
Error message: methodtype must be one of 훺𑏤¬ï|, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0004703d¨\U0006a9d5½\U000b3897\U000f572adJ°sñ\U0003deddÝ\x17\U00015bed\U000c0d61¬n\x1b='
Error message: methodtype must be one of 񇀽¨񪧕½򳢗󵜪dJ°sñ𽻝Ý𕯭󀵡¬n=, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000fc075\x19a'
Error message: methodtype must be one of 󼁵a, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x13\x9eº¨'
Error message: methodtype must be one of º¨, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ö\U00107a78¦÷Ê\U000927fd\x978©Ó¢'
Error message: methodtype must be one of ö􇩸¦÷Ê򒟽8©Ó¢, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '_\x88'
Error message: methodtype must be one of _, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'SL\x0e\x89'
Error message: methodtype must be one of SL, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'm¸\U0008b208j'
Error message: methodtype must be one of m¸򋈈j, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '6'
Error message: methodtype must be one of 6, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'æÑ5\U00052a3dø'
Error message: methodtype must be one of æÑ5񒨽ø, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x0b'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U00070a0aÓp'
Error message: methodtype must be one of 񰨊Óp, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x8d'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x85¨xrî¦\x1b¨\U0005ce0fa\U000631b1÷fÉ\U000e3dddq\xad7\U00090903R:\U0009c1f0é\U000952d8±\U0005bb28\U000acd36'
Error message: methodtype must be one of ¨xrî¦¨񜸏a񣆱÷fÉ󣷝q­7򐤃R:򜇰é򕋘±񛬨򬴶, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x1cÿD ¢\U0005f37b'
Error message: methodtype must be one of ÿD ¢񟍻, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x08'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0008dc23>i\U0008aafb\r\x83\x03V\U00076bb5N'
Error message: methodtype must be one of 򍰣>i򊫻V񶮵N, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '🏻👍🏻'
Error message: methodtype must be one of 🏻👍🏻, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'b+\U0010ffd9b'
Error message: methodtype must be one of b+􏿙b, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'þ'
Error message: methodtype must be one of þ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'TRUE'
Error message: methodtype must be one of TRUE, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x97\U00038229'
Error message: methodtype must be one of 𸈩, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x84y8.'
Error message: methodtype must be one of y8., got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '¢h£·Ì°\U000ea9a7å'
Error message: methodtype must be one of ¢h£·Ì°󪦧å, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '?\U0006abcbgy\U000bed7b'
Error message: methodtype must be one of ?񪯋gy򾵻, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '?\U0006abcbgy\U000bed7b'
Error message: methodtype must be one of ?񪯋gy򾵻, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Ö'
Error message: methodtype must be one of Ö, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 's\x98Jr\U000bf518L'
Error message: methodtype must be one of sJr򿔘L, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x88'
Error message: methodtype must be one of , got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'mÞId'
Error message: methodtype must be one of mÞId, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '0®ýÒ'
Error message: methodtype must be one of 0®ýÒ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ôW'
Error message: methodtype must be one of ôW, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0007c25e¦\U00084569\x94g\U000dcb7e'
Error message: methodtype must be one of 񼉞¦򄕩g󜭾, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '«\x99\x1b\U000dd6fd\U00066403à\U001060baDà\U000afaa0\x92𠓷\U000e4efc\U000c91d6Ì'
Error message: methodtype must be one of «󝛽񦐃à􆂺Dà򯪠𠓷󤻼󉇖Ì, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'è¶\U000ed6f8'
Error message: methodtype must be one of è¶󭛸, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U00019bd8'
Error message: methodtype must be one of 𙯘, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ù\U000a8947\\ó\x1c'
Error message: methodtype must be one of ù򨥇\ó, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: "'\x85"
Error message: methodtype must be one of ', got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\\ùD\x91vz\U000ef60a'
Error message: methodtype must be one of \ùDvz󯘊, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: ':\x96Áö¦'
Error message: methodtype must be one of :Áö¦, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ü~'
Error message: methodtype must be one of ü~, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x01𐐺.\x9b\x1eo¹H¡\U000678d1\U00071831Á\U00063e47äÚ-¿!Ñ\x0e\U000b8cd1\U000aa83e\x7fÚ\x92筆é'
Error message: methodtype must be one of 𐐺.o¹H¡񧣑񱠱Á񣹇äÚ-¿!Ñ򸳑򪠾Ú筆é, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '§\U0008fec97\x0c \U00094a84'
Error message: methodtype must be one of §򏻉7 򔪄, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\xa0\U0010e356\x07'
Error message: methodtype must be one of  􎍖, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '*\x1f\x0e\x13'
Error message: methodtype must be one of *, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Ü!|IË\x16'
Error message: methodtype must be one of Ü!|IË, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ú\x7fóg$öµ泪\x89j\x07ý½o\x91\U0010bb8b'
Error message: methodtype must be one of úóg$öµ泪jý½o􋮋, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ãã+\x99L'
Error message: methodtype must be one of ãã+L, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'w𭖢q\U00047895\U000c08be'
Error message: methodtype must be one of w𭖢q񇢕󀢾, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000436f8\x92'
Error message: methodtype must be one of 񃛸, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '±Æ}\x94\x0fêþ'
Error message: methodtype must be one of ±Æ}êþ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Æ\x9cÑø\U0004b87f'
Error message: methodtype must be one of ÆÑø񋡿, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'G\U00080472'
Error message: methodtype must be one of G򀑲, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0009d6e2Õ\U0009251e\U000710b8*\x06vK.\x04Æ\U0010c188Eï\x0c\U001051ed÷\U0003ceab'
Error message: methodtype must be one of 򝛢Õ򒔞񱂸*vK.Æ􌆈Eï􅇭÷𼺫, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: ' ¥\U000a0e4a'
Error message: methodtype must be one of  ¥򠹊, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x8e𝜴'
Error message: methodtype must be one of 𝜴, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x86Ñ'
Error message: methodtype must be one of Ñ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ª'
Error message: methodtype must be one of ª, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\ueaa2å\U0001c3f1\x06\xa01'
Error message: methodtype must be one of å𜏱 1, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U000e1028\x1c\U0001a8ce'
Error message: methodtype must be one of 󡀨𚣎, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U0004e7b0¨\U000397e1\x90\x9c\x0e'
Error message: methodtype must be one of 񎞰¨𹟡, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '2'
Error message: methodtype must be one of 2, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'Ä'
Error message: methodtype must be one of Ä, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\x000'
Error message: methodtype must be one of  0, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '¤è\U000bb311𬋯'
Error message: methodtype must be one of ¤è򻌑𬋯, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\U00103e94\x87\x89®'
Error message: methodtype must be one of 􃺔®, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '¤»\x96<G'
Error message: methodtype must be one of ¤»<G, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '\\¸\x7f'
Error message: methodtype must be one of \¸, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: '&\x14H\x9e¶\U000a1a19'
Error message: methodtype must be one of &H¶򡨙, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'ñ'
Error message: methodtype must be one of ñ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
Input: 'v\x0fÍ'
Error message: methodtype must be one of vÍ, got {'property', 'classmethod', 'method', 'staticmethod'} instead.
PASSED

============================== 1 passed in 0.38s ===============================
```
</details>

## Reproducing the Bug

```python
import pandas.errors


class DummyClass:
    pass


instance = DummyClass()

try:
    pandas.errors.AbstractMethodError(instance, methodtype='invalid_type')
except ValueError as e:
    print(str(e))
```

<details>

<summary>
ValueError: methodtype must be one of invalid_type, got {'method', 'property', 'classmethod', 'staticmethod'} instead.
</summary>
```
methodtype must be one of invalid_type, got {'method', 'property', 'classmethod', 'staticmethod'} instead.
```
</details>

## Why This Is A Bug

The error message format violates the universal convention for validation errors, which follows the pattern "expected X, got Y". The current implementation has the f-string template variables reversed, producing a message that says "methodtype must be one of invalid_type" when it should say "methodtype must be one of {'method', 'classmethod', 'staticmethod', 'property'}".

This makes the error message actively misleading and confusing for developers. When a developer passes an invalid methodtype, they need to know what the valid options are, but the current message shows their invalid input where the valid options should be listed. While the validation logic works correctly (invalid methodtypes are properly rejected), the swapped variables in the error message format can cause unnecessary debugging time and confusion.

## Relevant Context

The bug is located in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/errors/__init__.py` at line 298. The AbstractMethodError class is used internally by pandas to provide better error messages than the standard NotImplementedError for abstract methods. The `methodtype` parameter accepts only four valid values: 'method', 'classmethod', 'staticmethod', and 'property'.

The pandas documentation shows examples of using this error with valid methodtype values but does not specify the exact format of the ValueError message for invalid inputs. However, error message conventions across the Python ecosystem consistently use the "expected X, got Y" format.

## Proposed Fix

```diff
--- a/pandas/errors/__init__.py
+++ b/pandas/errors/__init__.py
@@ -295,7 +295,7 @@ class AbstractMethodError(NotImplementedError):
         types = {"method", "classmethod", "staticmethod", "property"}
         if methodtype not in types:
             raise ValueError(
-                f"methodtype must be one of {methodtype}, got {types} instead."
+                f"methodtype must be one of {types}, got {methodtype} instead."
             )
         self.methodtype = methodtype
```