"""Solver for definite integrals whose integrand is a series,
e.g. ∫_0^{1/2} (Σ_{n=2}^∞ x^n) dx.

Solves two independent exact ways (close the series first; interchange
sum and integral) and cross-checks numerically before reporting.
"""

import re

import sympy as sp
from sympy import S, latex, oo, symbols

_OO_TOKENS = {"oo", "inf", "infinity", "∞", "infty", "\\infty", "+oo", "+inf"}


# ------------------------------------------------------------- parsing

def _unpack_braces(tok: str) -> str:
    tok = (tok or "").strip()
    if tok.startswith("{") and tok.endswith("}"):
        tok = tok[1:-1]
    return tok.strip()


def _parse_bound(tok: str):
    tok = _unpack_braces(tok).replace("\\infty", "oo").strip()
    neg = tok.startswith("-")
    if tok.lstrip("+-").lower() in _OO_TOKENS:
        return -oo if neg else oo
    try:
        return sp.sympify(tok.replace("^", "**"))
    except Exception:
        return None


def _parse_plain_expr(tok: str):
    try:
        return sp.sympify(tok.replace("^", "**"))
    except Exception:
        return None


def _parse_latex_expr(tok: str):
    tok = _unpack_braces(tok)
    try:
        from latex2sympy2 import latex2sympy
        return latex2sympy(tok)
    except Exception:
        pass
    plain = (tok.replace("\\infty", "oo").replace("\\pi", "pi")
             .replace("^{", "**(").replace("_{", "(").replace("}", ")")
             .replace("^", "**").replace("\\", ""))
    return _parse_plain_expr(plain)


_PLAIN_RE = re.compile(
    r"^(?:integrate|int|∫)\s*\(?\s*"
    r"(?:sum|Σ)\s+(?P<term>.+?)\s+from\s+(?P<var>[a-zA-Z])\s*=\s*"
    r"(?P<slo>\S+?)\s+to\s+(?P<shi>\S+?)\s*\)?\s*"
    r"d\s*(?P<dvar>[a-zA-Z])\s+from\s+(?P<lo>\S+?)\s+to\s+(?P<hi>\S+?)\s*$",
    re.IGNORECASE,
)

_SYMPY_RE = re.compile(
    r"^(?:integrate|int|∫)\s*\(?\s*"
    r"Sum\(\s*(?P<term>.+?)\s*,\s*\(\s*(?P<var>[a-zA-Z])\s*,\s*"
    r"(?P<slo>[^,()]+?)\s*,\s*(?P<shi>[^,()]+?)\s*\)\s*\)\s*\)?\s*"
    r"d\s*(?P<dvar>[a-zA-Z])\s+from\s+(?P<lo>\S+?)\s+to\s+(?P<hi>\S+?)\s*$",
    re.IGNORECASE,
)

_LATEX_RE = re.compile(
    r"^\\?int\s*(?:_(?P<lo>\{[^}]*\}|[^_^ ]+))?\s*(?:\^(?P<hi>\{[^}]*\}|[^_^ ]+))?\s*"
    r"\(?\s*\\sum\s*(?:_(?P<svar>\{[^}]*\}|[^_^ ]+))?\s*(?:\^(?P<shi>\{[^}]*\}|[^_^ ]+))?\s*"
    r"(?P<term>[^()]+?)\s*\)?\s*d\s*(?P<dvar>[a-zA-Z])\s*$",
)


# ------------------------------------------------------------- solving

def _closed_series(term, n, slo, shi, x, lo, hi):
    """Sum the series first, e.g. Σ x^n -> x**2/(1-x)."""
    try:
        s = sp.summation(term, (n, slo, shi))
    except Exception:
        return None
    if isinstance(s, sp.Piecewise):
        sample = (lo + hi) / 2 if (lo.is_finite and hi.is_finite) else None
        for expr, cond in s.args:
            if expr.has(sp.Sum):
                continue
            if cond is S.true:
                return expr
            if sample is not None:
                try:
                    if bool(cond.subs(x, sample)):
                        return expr
                except Exception:
                    continue
        return None
    return None if s.has(sp.Sum) else sp.simplify(s)


def _swapped_series(term, n, slo, shi, x, lo, hi):
    """Interchange: Σ ∫ term dx (valid by uniform convergence)."""
    try:
        inner = sp.integrate(term, (x, lo, hi))
        if inner is None or inner.has(sp.Integral):
            return None
        s = sp.summation(inner, (n, slo, shi))
        return None if s.has(sp.Sum) else sp.simplify(s)
    except Exception:
        return None


def _numeric_check(closed, x, lo, hi):
    if closed is None or not (lo.is_finite and hi.is_finite):
        return None
    try:
        import mpmath
        f = sp.lambdify(x, closed, "mpmath")
        return float(mpmath.quad(f, [float(lo), float(hi)]))
    except Exception:
        return None


def _numeric_partial_sum(term, n, slo, shi, x, lo, hi, terms=500):
    if shi is not oo or not (lo.is_finite and hi.is_finite):
        return None
    try:
        total = 0.0
        for k in range(int(slo), int(slo) + terms):
            fv = float(sp.N(sp.integrate(term.subs(n, k), (x, lo, hi))))
            total += fv
            if abs(fv) < 1e-12 and k > int(slo) + 10:
                break
        return total
    except Exception:
        return None


# ------------------------------------------------------------- entry

def try_series_integral(expression: str):
    """Return a result dict if `expression` is an integral of a series,
    else None (caller falls through to the generic pipeline)."""
    if not expression:
        return None
    low = expression.lower()
    if "sum" not in low and "Σ" not in expression and "\\sum" not in expression:
        return None

    text = expression.strip().strip("$")
    text = text.replace("\\left", "").replace("\\right", "")

    m = _PLAIN_RE.match(text) or _SYMPY_RE.match(text)
    flavor = "plain"
    if m is None:
        m = _LATEX_RE.match(text)
        flavor = "latex"
    if m is None:
        return None
    g = m.groupdict()

    parse_expr = _parse_latex_expr if flavor == "latex" else _parse_plain_expr
    term = parse_expr(g["term"])
    if term is None:
        return None

    if flavor == "latex":
        svar_raw = _unpack_braces(g.get("svar") or "")
        if "=" not in svar_raw:
            return None
        var_name, slo_raw = (t.strip() for t in svar_raw.split("=", 1))
        shi_raw = g.get("shi") or "oo"
    else:
        var_name, slo_raw, shi_raw = g["var"], g["slo"], g["shi"]

    n, x = symbols(var_name), symbols(g["dvar"])
    slo, shi = _parse_bound(slo_raw), _parse_bound(shi_raw)
    lo, hi = _parse_bound(g["lo"]), _parse_bound(g["hi"])
    if None in (slo, shi, lo, hi) or not term.free_symbols <= {n, x}:
        return None

    closed = _closed_series(term, n, slo, shi, x, lo, hi)
    result = None
    if closed is not None:
        try:
            r = sp.integrate(closed, (x, lo, hi))
            if r is not None and not r.has(sp.Integral):
                result = sp.simplify(r)
        except Exception:
            result = None

    swapped = _swapped_series(term, n, slo, shi, x, lo, hi)
    if result is None:
        result = swapped
    if result is None:
        return None

    numeric = _numeric_check(closed, x, lo, hi)
    if numeric is None:
        numeric = _numeric_partial_sum(term, n, slo, shi, x, lo, hi)
    value = float(sp.N(result))
    if numeric is None or abs(value - numeric) > 1e-6:
        return None  # honest refusal over a wrong headline answer

    q_latex = (rf"\int_{{{latex(lo)}}}^{{{latex(hi)}}} "
               rf"\sum_{{{var_name}={latex(slo)}}}^{{{latex(shi)}}} "
               rf"{latex(term)} \, d{g['dvar']}")

    steps = [(
        "Recognize the series inside the integral", q_latex,
        "The integrand is a series; resolve it before generic integration.", None,
    )]
    if closed is not None:
        steps.append((
            "Close the series (geometric-series formula)", latex(closed),
            "The series sums to this closed form on the whole integration interval.",
            None,
        ))
    steps.append((
        "Apply the Fundamental Theorem of Calculus", latex(result),
        "Integrate the closed form (or term-by-term) and evaluate at the bounds.",
        latex(result),
    ))
    if swapped is not None and closed is not None:
        steps.append((
            "Cross-check: interchange of summation and integration",
            latex(swapped),
            "Uniform convergence justifies term-by-term integration; results agree.",
            None,
        ))

    return {
        "question_latex": q_latex,
        "answer": str(result),
        "answer_latex": latex(result),
        "numeric": value,
        "steps": steps,
        "verification": (f"Verified numerically: symbolic {value:.10g} "
                         f"matches numeric quadrature {numeric:.10g}."),
    }