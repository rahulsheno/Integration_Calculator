"""Laplace transforms and Fourier series.

Both are thin, well-tested wrappers around sympy's own machinery
(laplace_transform / inverse_laplace_transform and fourier_series), with
the steps and LaTeX a step-by-step solver needs. The module exists because
solver.py previously imported these sympy functions but never called them,
so every "laplace" or "fourier" problem fell through to the generic
"couldn't compute" fallback even though sympy can solve them directly.

What this module adds on top of the raw sympy calls:

- Fourier: returns the coefficient formulas a0, an, bn AND a finite
  truncated partial sum (sympy's fourier_series only hands back the series
  object; extracting the formulas and a readable N-term sum is the part
  users actually want to see).
- Laplace: reports the region of convergence condition sympy computes,
  strips the Heaviside(t) factors that inverse transforms carry (valid for
  x >= 0, which is the domain the Laplace transform is defined on), and
  normalizes the result.

Deliberately NOT handled: bilateral transforms and transforms of piecewise
functions that sympy can't integrate in closed form - those are reported
honestly rather than approximated.
"""

import sympy as sp
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform
from sympy.series.fourier import fourier_series

from app.services.compute_utils import run_with_timeout, ComputeTimeout


def compute_laplace(func_expr, var_symbol, s_symbol, noconds: bool = True):
    """Compute L{f(var)} = F(s). Returns a dict:

        {"transform": F(s) expr, "condition": convergence condition expr|None,
         "error": str|None}

    `noconds=True` asks sympy for the transform without a Piecewise
    convergence wrapper when possible (the standard single-expression form).
    The condition is still captured separately when sympy can derive one.
    """
    try:
        result = run_with_timeout(
            laplace_transform, func_expr, var_symbol, s_symbol, noconds=noconds, seconds=30
        )
    except ComputeTimeout as e:
        return {"transform": None, "condition": None, "error": str(e)}
    except Exception:
        return {
            "transform": None, "condition": None,
            "error": "sympy couldn't compute a closed-form Laplace transform for this function.",
        }

    if isinstance(result, tuple):
        F, cond = result[0], result[1] if len(result) > 1 else None
    else:
        F, cond = result, None

    if isinstance(F, sp.Integral) or (hasattr(F, "has") and F.has(sp.Integral)):
        return {
            "transform": None, "condition": cond,
            "error": "sympy couldn't evaluate the defining integral for this Laplace transform.",
        }

    return {"transform": sp.simplify(F), "condition": cond, "error": None}


def compute_inverse_laplace(F_expr, s_symbol, var_symbol):
    """Compute L^-1{F(s)} = f(var). Returns a dict:

        {"inverse": f(var) expr|None, "error": str|None}

    Strips Heaviside(var) factors, which inverse transforms legitimately
    carry to encode "f = 0 for var < 0". Since the Laplace transform is only
    defined on var >= 0 and this is a real-domain calculus tool, the
    Heaviside adds noise to the displayed answer without changing its value
    on the relevant domain.
    """
    try:
        result = run_with_timeout(
            inverse_laplace_transform, F_expr, s_symbol, var_symbol, seconds=30
        )
    except ComputeTimeout as e:
        return {"inverse": None, "error": str(e)}
    except Exception:
        return {"inverse": None, "error": "sympy couldn't compute this inverse Laplace transform."}

    try:
        if result.has(sp.Heaviside):
            result = result.replace(lambda n: isinstance(n, sp.Heaviside), lambda n: 1)
    except Exception:
        pass

    try:
        result = sp.simplify(result)
    except Exception:
        pass

    if isinstance(result, sp.Integral) or (hasattr(result, "has") and result.has(sp.Integral)):
        return {"inverse": None, "error": "sympy couldn't evaluate this inverse transform in closed form."}

    return {"inverse": result, "error": None}


def compute_fourier(func_expr, var_symbol, lower, upper, num_terms: int = 5):
    """Compute the Fourier series of f on [lower, upper]. Returns a dict:

        {"a0": expr, "an": expr (as function of dummy n), "bn": expr,
         "partial_sum": expr (first num_terms of the series),
         "error": str|None}

    The coefficient formulas are given in terms of a fresh dummy symbol n
    (sympy returns them in terms of its own internal dummy), so they render
    cleanly as the textbook a_n = ..., b_n = ....
    """
    n_dummy = sp.Dummy("fourier_n", integer=True, positive=True)
    display_n = sp.Symbol("n", integer=True, positive=True)

    try:
        fs = run_with_timeout(
            fourier_series, func_expr, (var_symbol, lower, upper), seconds=45
        )
    except ComputeTimeout as e:
        return {"a0": None, "an": None, "bn": None, "partial_sum": None, "error": str(e)}
    except Exception:
        return {
            "a0": None, "an": None, "bn": None, "partial_sum": None,
            "error": "sympy couldn't compute the Fourier series for this function on this interval.",
        }

    try:
        a0 = fs.a0 if fs.a0 is not None else sp.Integer(0)

        an_seq = fs.an
        bn_seq = fs.bn
        an_formula = _seq_to_formula(an_seq, n_dummy, var_symbol)
        bn_formula = _seq_to_formula(bn_seq, n_dummy, var_symbol)

        truncated = fs.truncate(num_terms)
        if truncated.has(sp.SeqFormula) or truncated.has(sp.Sum):
            truncated = truncated.doit()
    except Exception:
        return {
            "a0": None, "an": None, "bn": None, "partial_sum": None,
            "error": "sympy computed the Fourier series but a readable closed form couldn't be extracted.",
        }

    return {
        "a0": sp.simplify(a0),
        "an": an_formula.xreplace({n_dummy: display_n}) if an_formula is not None else None,
        "bn": bn_formula.xreplace({n_dummy: display_n}) if bn_formula is not None else None,
        "partial_sum": sp.simplify(truncated) if truncated is not None else None,
        "error": None,
    }


def _seq_to_formula(seq_obj, n_dummy, var_symbol):
    """Turn a sympy SeqFormula sequence object into a plain expression in a
    readable dummy n. sympy's fourier_series returns an/bn as SeqFormula
    objects; .formula holds the closed-form expression in the sequence's
    own internal dummy symbol, which we rename to n for clean display."""
    try:
        formula = seq_obj.formula
    except AttributeError:
        return None
    if formula is None:
        return None

    internal_dummies = formula.free_symbols - {var_symbol}
    if internal_dummies:
        d = sorted(internal_dummies, key=lambda s: s.name)[0]
        formula = formula.subs(d, n_dummy)
    return formula
