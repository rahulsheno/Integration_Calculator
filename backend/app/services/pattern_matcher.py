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


def _is_finite_bound(val) -> bool:
    """True only for a genuinely finite numeric bound. Both techniques below
    assume a bounded interval [a, b] - King's rule reflects x -> a+b-x (which
    degenerates to a meaningless x -> ∞-x, i.e. just ∞ again, when either
    bound is infinite), and the periodicity count divides the interval's
    span by the period (which is itself infinite, or produces an infinite
    period count, over an unbounded interval). Rather than let sympy grind
    through a substitution or division it can't meaningfully do and hand
    back "nan" (or an infinite loop's worth of periods) as if it were a
    real, verified answer, bail out up front whenever a bound isn't finite.
    """
    try:
        return bool(val.is_finite)
    except Exception:
        return False


def try_symmetry_substitution(expr, var, lower_val, upper_val):
    """Attempt the a+b-x symmetry substitution ("King's rule"). Returns the
    resulting definite integral value, or None if the substitution doesn't
    simplify things (i.e. isn't worth using) or can't be evaluated.
    """
    if not (_is_finite_bound(lower_val) and _is_finite_bound(upper_val)):
        # x -> a+b-x only makes sense as a reflection of a bounded interval
        # onto itself. With an infinite bound (e.g. upper_val = oo), a+b-x
        # is just ∞-x, which sympy simplifies straight back to ∞ regardless
        # of x - substituting that in doesn't reflect anything, and
        # previously produced "nan" for the whole combined expression
        # (confirmed: log(tanh(x/2)) over [0, oo) silently returned "nan"
        # as if it were a verified answer, via exactly this path).
        return None

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

    result = sp.simplify(half_integral / 2)
    # Belt-and-suspenders: even with the finite-bounds guard above, if
    # sympy's own simplification chain ever produces nan/complex-infinity
    # some other way, don't hand that back as if it were a real result.
    if result in (sp.nan, sp.zoo) or (hasattr(result, "has") and result.has(sp.nan)):
        return None
    return result


def try_periodicity_reduction(expr, var, lower_val, upper_val):
    """If expr is periodic with period T and [lower_val, upper_val] spans
    complete periods, reduce to (count * integral over one period) plus the
    integral over any partial remainder - potentially much cheaper/more
    tractable than integrating the full span directly. Returns the result,
    or None if the integrand isn't (detectably) periodic, the interval
    doesn't span at least 2 full periods (not worth it for fewer), or
    evaluation fails.
    """
    if not (_is_finite_bound(lower_val) and _is_finite_bound(upper_val)):
        # "How many periods fit in [a, b]" is meaningless once the interval
        # itself is unbounded - span/period is infinite, so num_periods
        # would be infinite too (an infinite loop's worth of periods, not a
        # real count), rather than the sympy exception this used to rely on
        # to bail out safely.
        return None

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
