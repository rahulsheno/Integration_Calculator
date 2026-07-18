"""Pattern recognition for definite integrals.

sympy's integrate() already solves most "named" integral families directly
and correctly (Gaussian, Beta, Gamma forms were all checked and work fine
natively - no fast path needed there). The two techniques in this module
exist because sympy genuinely can't do them on its own:

- Symmetry substitution ("King's rule"): for integral(f, (x, a, b)), the
  substitution x -> a+b-x leaves the integral's VALUE unchanged (since it's
  just relabeling the same area), so integral(f) = integral(f(a+b-x)). Adding
  these two equal integrals gives 2*integral(f) = integral(f(x) + f(a+b-x)).
  When f(x) + f(a+b-x) simplifies to something much simpler than f alone
  (the classic case), this solves integrals sympy's direct approach fails on
  entirely - verified against a textbook example (x*sin(x)/(1+cos(x)**2)
  over [0, pi], sympy returns it unevaluated; this technique correctly gets
  pi**2/4) before being trusted here.

- Periodicity reduction: if the integrand is periodic and the bounds span
  many complete periods, the integral equals (number of complete periods) *
  (integral over one period) + (integral over the remainder). This can turn
  an expensive or intractable large-interval integral into a cheap one.

Every result from this module should be treated as a candidate, not a final
answer - callers are expected to numerically verify it (as solver.py's
_numerical_verify already does for every definite integral) before
presenting it.
"""

import sympy as sp
from sympy.calculus.util import periodicity


def try_symmetry_substitution(expr, var, lower_val, upper_val):
    """Attempt the a+b-x symmetry substitution ("King's rule"). Returns the
    resulting definite integral value, or None if the substitution doesn't
    simplify things (i.e. isn't worth using) or can't be evaluated.
    """
    try:
        reflected = expr.subs(var, lower_val + upper_val - var)
        combined = sp.simplify(expr + reflected)
    except Exception:
        return None

    if combined is None:
        return None

    # Only worth trying if this genuinely simplified things - otherwise
    # we're just doing extra work for no benefit (or even picking up a
    # harder expression than the original).
    try:
        if sp.count_ops(combined) >= sp.count_ops(expr) * 2:
            return None
    except Exception:
        pass

    try:
        half_integral = sp.integrate(combined, (var, lower_val, upper_val))
    except Exception:
        return None

    if half_integral is None or isinstance(half_integral, sp.Integral) or half_integral.has(sp.Integral):
        return None

    return sp.simplify(half_integral / 2)


def try_periodicity_reduction(expr, var, lower_val, upper_val):
    """If expr is periodic with period T and [lower_val, upper_val] spans
    complete periods, reduce to (count * integral over one period) plus the
    integral over any partial remainder - potentially much cheaper/more
    tractable than integrating the full span directly. Returns the result,
    or None if the integrand isn't (detectably) periodic, the interval
    doesn't span at least 2 full periods (not worth it for fewer), or
    evaluation fails.
    """
    try:
        period = periodicity(expr, var)
    except Exception:
        period = None

    if period is None or period == 0:
        return None

    try:
        span = sp.Abs(upper_val - lower_val)
        period = sp.Abs(period)
        num_periods = sp.floor(span / period)
        if num_periods < 2:
            return None

        remainder = span - num_periods * period
        one_period_integral = sp.integrate(expr, (var, lower_val, lower_val + period))
        if one_period_integral is None or one_period_integral.has(sp.Integral):
            return None

        total = num_periods * one_period_integral
        if remainder != 0:
            remainder_integral = sp.integrate(expr, (var, lower_val + num_periods * period, upper_val))
            if remainder_integral is None or remainder_integral.has(sp.Integral):
                return None
            total += remainder_integral

        return sp.simplify(total), period, num_periods
    except Exception:
        return None


def classify_named_pattern(expr, var, lower_val=None, upper_val=None) -> str | None:
    """Best-effort label for well-known integral families, purely for nicer
    step explanations - never affects the actual computed value.
    """
    for exp_atom in expr.atoms(sp.exp):
        arg = exp_atom.args[0]
        if arg == -var**2:
            return "Gaussian Integral"
        if arg.is_Mul:
            factors = arg.args
            if any(f == var**2 for f in factors) and any(f.is_number and f.is_negative for f in factors):
                return "Gaussian Integral"

    if lower_val == 0 and upper_val == 1 and expr.is_Mul:
        factors = expr.as_ordered_factors()
        has_x_pow = any(f.is_Pow and f.base == var for f in factors)
        has_complement_pow = any(f.is_Pow and sp.simplify(f.base - (1 - var)) == 0 for f in factors)
        if has_x_pow and has_complement_pow:
            return "Beta Integral"

    if lower_val == 0 and upper_val is sp.oo and expr.has(sp.exp(-var)):
        return "Gamma Integral"

    return None
