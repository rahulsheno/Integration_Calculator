"""Infinite series and continued fraction recognition.

Handles two specific, well-defined classes of "infinite" expression that
sympy's ordinary integrate()/simplify() pipeline never sees on its own,
because nothing in the normal flow ever calls Sum(...).doit() or sets up a
self-referential equation:

- Infinite series: sum(f(n), (n, a, oo)) - delegated to sympy's own Sum.doit(),
  which already has closed forms for geometric series, p-series/zeta values,
  and many other standard forms. The value this module adds is detecting
  that a piece of text IS a series in the first place, and reporting the
  recognized pattern (geometric, p-series, ...) alongside the result.

- Periodic (self-similar) continued fractions written with a literal "..."
  marking the repetition, e.g. "1/(1+x/(1+x/(1+...)))". These are solved by
  substituting a fresh symbol for the self-similar tail and solving the
  resulting algebraic equation - not by literally expanding the recursion,
  which would blow up arbitrarily (item 12's concern).

Deliberately NOT handled: telescoping series with no closed algebraic form
sympy already knows, arbitrary (non-periodic) continued fractions, and
series given only as a handful of numeric terms with no explicit formula
(genuine sequence inference is a much harder, separate problem).
"""

import re
import sympy as sp


def try_solve_product(term_str: str, var_name: str, lower_str: str, upper_str: str, safe_sympify):
    """Attempt to evaluate an infinite (or finite) product prod(term, (var, lower, upper)).
    Mirrors try_solve_series - same symbol-matching fix applies (see there).
    """
    term, err = safe_sympify(term_str)
    if term is None:
        return None

    name_matches = [s for s in term.free_symbols if s.name == var_name]
    var = name_matches[0] if name_matches else sp.Symbol(var_name)

    lower_expr, _ = safe_sympify(lower_str)
    if lower_expr is None:
        return None

    upper_norm = upper_str.strip().lower()
    if upper_norm in ("oo", "infinity", "inf", "∞"):
        upper_expr = sp.oo
    else:
        upper_expr, _ = safe_sympify(upper_str)
        if upper_expr is None:
            return None

    product_expr = sp.Product(term, (var, lower_expr, upper_expr))
    try:
        result = product_expr.doit()
    except Exception:
        result = product_expr

    return {
        "term": term, "var": var, "lower": lower_expr, "upper": upper_expr,
        "product_expr": product_expr, "result": result,
        "evaluated": not isinstance(result, sp.Product),
    }


def try_solve_series(term_str: str, var_name: str, lower_str: str, upper_str: str, safe_sympify):
    """Attempt to evaluate an infinite (or finite) series sum(term, (var, lower, upper)).

    Returns a dict with the parsed pieces and result on success, or None if
    the term/bounds couldn't be parsed at all. The result itself may be an
    unevaluated Sum if sympy has no closed form - callers should check for
    that rather than assume a numeric/closed-form answer.
    """
    term, err = safe_sympify(term_str)
    if term is None:
        return None

    # Use the actual symbol instance embedded in the parsed term - it may
    # carry assumptions (e.g. integer=True) from safe_sympify's own symbol
    # table, and a freshly created plain Symbol(var_name) here would compare
    # unequal to it, silently summing over an unrelated "n" that never
    # actually appears in term (sympy would then treat the whole term as a
    # constant with respect to the mismatched variable, giving nonsense like
    # oo*x**n instead of a real geometric-series closed form).
    name_matches = [s for s in term.free_symbols if s.name == var_name]
    var = name_matches[0] if name_matches else sp.Symbol(var_name)

    lower_expr, _ = safe_sympify(lower_str)
    if lower_expr is None:
        return None

    upper_norm = upper_str.strip().lower()
    if upper_norm in ("oo", "infinity", "inf", "∞"):
        upper_expr = sp.oo
    else:
        upper_expr, _ = safe_sympify(upper_str)
        if upper_expr is None:
            return None

    sum_expr = sp.Sum(term, (var, lower_expr, upper_expr))

    try:
        result = sum_expr.doit()
    except Exception:
        result = sum_expr

    pattern_name = _classify_series_pattern(term, var)

    return {
        "term": term,
        "var": var,
        "lower": lower_expr,
        "upper": upper_expr,
        "sum_expr": sum_expr,
        "result": result,
        "pattern": pattern_name,
        # "Evaluated" means sympy's doit() found an actual closed form,
        # possibly conditional (e.g. a Piecewise valid for |x|<1) - not that
        # every branch is Sum-free, since a divergent-case branch may still
        # display the unevaluated Sum for that condition.
        "evaluated": not isinstance(result, sp.Sum),
    }


def _classify_series_pattern(term, var) -> str | None:
    """Best-effort label for the recognized series family, for nicer step
    explanations. Returns None if it doesn't match a named pattern - the
    series can still be evaluated via sympy even without a label.
    """
    # Geometric: term = c * r**n (possibly with n appearing only in the exponent)
    if term.is_Mul or term.is_Pow or term.is_Symbol:
        n_in_exponent = any(
            isinstance(a, sp.Pow) and var in a.exp.free_symbols and var not in a.base.free_symbols
            for a in sp.preorder_traversal(term)
            if isinstance(a, sp.Pow)
        )
        if n_in_exponent:
            return "Geometric Series"

    # p-series: term = 1/n**p for constant p
    if term.is_Pow and term.base == var and (-term.exp).is_number:
        return "p-Series"
    if term.is_Pow and term.exp.is_number and term.exp.is_negative and term.base == var:
        return "p-Series"

    denom = sp.denom(sp.together(term))
    if denom.is_Pow and denom.base == var and denom.exp.is_number:
        return "p-Series"

    return None


_CF_PLACEHOLDER = "W"  # single letter, and avoids sympy's reserved names (Q, E, I, S, N, O)


def try_solve_continued_fraction(text: str, safe_sympify):
    """Detect a periodic infinite continued fraction written with a literal
    "..." marking the repetition (e.g. "1/(1+x/(1+x/(1+...)))"), and solve
    it algebraically.

    The key subtlety: the truly self-similar quantity is the *repeating
    unit* applied to itself, not the whole wrapped expression. For
    "1/(1+x/(1+x/(1+...)))", the repeating unit is "1+x/(...)" - call its
    value W. W satisfies W = 1+x/W (itself, one level in), and the final
    answer is prefix applied once to W: 1/(W). Treating the *entire*
    wrapped string (including the outer "1/(" that does NOT itself repeat)
    as self-referential gives a different, wrong equation - this was
    checked against direct high-depth numeric recursion of the actual
    nested fraction before being trusted here.

    Only proceeds if the same bracketed unit is found repeated at least
    twice before the "..." - the safety gate that keeps this from guessing
    at expressions that merely contain "..." without actually being
    periodic. Returns None if that gate isn't met, parsing fails, or the
    equation has no solution.
    """
    if "..." not in text:
        return None

    compact = text.replace(" ", "")
    idx = compact.index("...")
    body = compact[:idx]

    segments = body.split("(")
    if len(segments) < 4:
        return None
    # Drop the outer prefix (segments[0]) and the last segment (likely
    # truncated exactly at the "..." cut, so not a fair comparison).
    inner_segments = segments[1:-1]
    if len(inner_segments) < 2 or len(set(inner_segments)) != 1:
        return None

    prefix, unit = segments[0], segments[1]

    unit_closed = unit + f"({_CF_PLACEHOLDER})"
    unit_expr, err = safe_sympify(unit_closed)
    if unit_expr is None:
        return None
    placeholder_matches = [s for s in unit_expr.free_symbols if s.name == _CF_PLACEHOLDER]
    if not placeholder_matches:
        return None
    w = placeholder_matches[0]
    w_positive = sp.Symbol(_CF_PLACEHOLDER, positive=True)
    unit_expr = unit_expr.subs(w, w_positive)
    w = w_positive

    try:
        w_solutions = sp.solve(sp.Eq(w, unit_expr), w)
    except Exception:
        return None
    if not w_solutions:
        return None

    other_vars = sorted((unit_expr.free_symbols - {w}), key=lambda s: s.name)
    sample_subs = {v: 2 for v in other_vars}

    # Independent ground truth: directly (numerically) iterate the ORIGINAL
    # nested structure many levels deep, rather than trusting the algebra
    # alone - this is what caught the original version of this function
    # solving the wrong equation (it validated against its own, differently
    # wrong, iteration instead of the actual nested fraction).
    numeric_truth = _iterate_original_structure(body, prefix, sample_subs, safe_sympify)

    candidates = []
    for w_sol in w_solutions:
        try:
            final_str = prefix + "(" + str(w_sol) + ")"
            final_expr, _ = safe_sympify(final_str)
        except Exception:
            continue
        if final_expr is None:
            continue
        try:
            final_numeric = complex(sp.N(final_expr.subs(sample_subs)))
        except Exception:
            continue
        matches_truth = (
            numeric_truth is not None and abs(final_numeric - numeric_truth) < 1e-4
        )
        candidates.append({"w_solution": w_sol, "final_expr": final_expr, "matches_truth": matches_truth})

    return {
        "unit_equation": sp.Eq(w, unit_expr),
        "symbol": w,
        "candidates": candidates,
        "numeric_truth": numeric_truth,
        "sample_subs": sample_subs,
    }


def _iterate_original_structure(body: str, prefix: str, sample_subs, safe_sympify, iterations: int = 200):
    """Ground truth for validation: repeatedly apply the repeating unit to a
    numeric seed to converge on the true value of the self-similar part,
    then apply prefix once - mirroring exactly how the nested fraction is
    actually built up, without ever constructing an exponentially large
    expression tree.
    """
    try:
        unit = body.split("(")[1]
        unit_expr, _ = safe_sympify(unit + f"({_CF_PLACEHOLDER})")
        if unit_expr is None:
            return None
        placeholder_matches = [s for s in unit_expr.free_symbols if s.name == _CF_PLACEHOLDER]
        if not placeholder_matches:
            return None
        w = placeholder_matches[0]
        unit_expr = unit_expr.subs(sample_subs)

        f = sp.lambdify(w, unit_expr, modules=["mpmath"])
        import mpmath as mp

        v = mp.mpf(1)
        for _ in range(iterations):
            v = mp.mpf(complex(f(v)).real)

        final_str = prefix + "(" + str(float(v)) + ")"
        final_expr, _ = safe_sympify(final_str)
        if final_expr is None:
            return None
        return complex(sp.N(final_expr.subs(sample_subs)))
    except Exception:
        return None
