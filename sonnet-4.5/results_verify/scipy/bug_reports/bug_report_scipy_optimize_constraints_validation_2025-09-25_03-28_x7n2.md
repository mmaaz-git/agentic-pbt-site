# Bug Report: scipy.optimize Constraint Classes Missing Input Validation

**Target**: `scipy.optimize.Bounds`, `scipy.optimize.LinearConstraint`, `scipy.optimize.NonlinearConstraint`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

All three constraint classes (`Bounds`, `LinearConstraint`, `NonlinearConstraint`) accept infeasible constraints where `lb > ub` without raising an error during construction, violating their documented contracts. This makes debugging harder as errors only appear later during optimization.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint


@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=5),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=5),
)
@settings(max_examples=200, deadline=2000)
def test_bounds_validates_lb_le_ub(lb_list, ub_list):
    lb = np.array(lb_list[:min(len(lb_list), len(ub_list))])
    ub = np.array(ub_list[:min(len(lb_list), len(ub_list))])

    bounds = Bounds(lb, ub)
    invalid_mask = bounds.lb > bounds.ub

    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        assert False, (
            f"Bounds allows infeasible constraint: "
            f"lb={bounds.lb}, ub={bounds.ub}. "
            f"At indices {invalid_indices}: lb > ub."
        )


@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=3),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=3),
)
@settings(max_examples=200, deadline=2000)
def test_linear_constraint_validates_lb_le_ub(lb_list, ub_list):
    n = min(len(lb_list), len(ub_list))
    lb = np.array(lb_list[:n])
    ub = np.array(ub_list[:n])
    A = np.eye(n)

    constraint = LinearConstraint(A, lb, ub)
    invalid_mask = np.array(constraint.lb) > np.array(constraint.ub)

    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        assert False, (
            f"LinearConstraint allows infeasible constraint at indices {invalid_indices}"
        )


@given(
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=3),
    st.lists(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=3),
)
@settings(max_examples=200, deadline=2000)
def test_nonlinear_constraint_validates_lb_le_ub(lb_list, ub_list):
    n = min(len(lb_list), len(ub_list))
    lb = lb_list[:n]
    ub = ub_list[:n]

    def identity(x):
        return x[:n]

    constraint = NonlinearConstraint(identity, lb, ub)
    lb_array = np.array(constraint.lb)
    ub_array = np.array(constraint.ub)
    invalid_mask = lb_array > ub_array

    if np.any(invalid_mask):
        invalid_indices = np.where(invalid_mask)[0]
        assert False, (
            f"NonlinearConstraint allows infeasible constraint at indices {invalid_indices}"
        )
```

**Failing inputs**:
- `Bounds`: `lb=[1.0], ub=[0.0]`
- `LinearConstraint`: `lb=[1.0], ub=[0.0]`
- `NonlinearConstraint`: `lb=[0.0], ub=[-1.0]`

## Reproducing the Bug

```python
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

lb = np.array([1.0])
ub = np.array([0.0])

bounds = Bounds(lb, ub)
print(f"Bounds created: lb={bounds.lb}, ub={bounds.ub}")
print(f"lb > ub: {bounds.lb[0] > bounds.ub[0]}")

A = np.array([[1.0]])
lc = LinearConstraint(A, lb, ub)
print(f"LinearConstraint created: lb={lc.lb}, ub={lc.ub}")
print(f"lb > ub: {lc.lb[0] > lc.ub[0]}")

def identity(x):
    return x

nlc = NonlinearConstraint(identity, lb.tolist(), ub.tolist())
print(f"NonlinearConstraint created: lb={nlc.lb}, ub={nlc.ub}")
print(f"lb > ub: {nlc.lb[0] > nlc.ub[0]}")

print("All three classes accepted infeasible constraints!")

def objective(x):
    return x[0] ** 2

try:
    result = minimize(objective, [0.5], bounds=bounds, method='L-BFGS-B')
except ValueError as e:
    print(f"minimize with Bounds raises: {e}")
```

Output:
```
Bounds created: lb=[1.], ub=[0.]
lb > ub: True
LinearConstraint created: lb=[1.], ub=[0.]
lb > ub: True
NonlinearConstraint created: lb=[1.0], ub=[0.0]
lb > ub: True
All three classes accepted infeasible constraints!
minimize with Bounds raises: An upper bound is less than the corresponding lower bound.
```

## Why This Is A Bug

All three classes document their constraints as:
- **Bounds**: `"lb <= x <= ub"`
- **LinearConstraint**: `"lb <= A.dot(x) <= ub"`
- **NonlinearConstraint**: `"lb <= fun(x) <= ub"`

When `lb > ub`, no value can satisfy these constraints, making them infeasible. These classes should validate this invariant at construction time.

**Current behavior:** All three classes accept `lb > ub` without error. Validation only occurs later when optimizers use these constraints.

**Expected behavior:** Constructors should raise `ValueError` immediately when `lb > ub` for any component.

**Impact:**
- Delayed error detection makes debugging harder
- Users may pass invalid constraint objects around before discovering errors
- Error messages appear in optimizers rather than at the source
- Violates fail-fast principle and constructor invariants
- Same bug across multiple classes suggests shared validation code issue

## Fix

All three classes likely share validation logic in `_constraints.py`. Add validation to the `_input_validation` method or similar:

```diff
--- a/scipy/optimize/_constraints.py
+++ b/scipy/optimize/_constraints.py
@@ -42,6 +42,10 @@ class Bounds:
     def _input_validation(self):
         try:
             res = np.broadcast_arrays(self.lb, self.ub, self.keep_feasible)
             self.lb, self.ub, self.keep_feasible = res
         except ValueError:
             message = "`lb`, `ub`, and `keep_feasible` must be broadcastable."
             raise ValueError(message)
+
+        if np.any(self.lb > self.ub):
+            invalid = np.where(self.lb > self.ub)[0]
+            raise ValueError(
+                f"Lower bounds must not exceed upper bounds. "
+                f"Violation at indices {invalid.tolist()}: "
+                f"lb={self.lb[invalid].tolist()} > ub={self.ub[invalid].tolist()}"
+            )
```

The same validation should be added to `LinearConstraint` and `NonlinearConstraint` classes.