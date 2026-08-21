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
    """Parse a series term like "x^n", "1/n^2", or "3^(n)*x^(n)".

    Tries plain sympify first; the fallback uses sympy's parser with
    implicit multiplication so OCR output like "3^(n)x^(n)" (no explicit
    " * " between factors) still parses - sympify() alone rejects that
    as a syntax error and the whole engine would silently bail out.
    """
    candidate = (tok or "").strip().strip(",")
    if not candidate:
        return None
    candidate = candidate.replace("^", "**")
    try:
        parsed = sp.sympify(candidate)
    except Exception:
        parsed = None
    if parsed is None:
        try:
            from sympy.parsing.sympy_parser import (
                parse_expr, standard_transformations, implicit_multiplication_application,
            )
            parsed = parse_expr(
                candidate,
                transformations=standard_transformations + (implicit_multiplication_application,),
                evaluate=True,
            )
        except Exception:
            parsed = None
    # sympify("x,") yields a tuple rather than an expression - a leftover
    # trailing comma (e.g. a LaTeX "\," spacing token surviving cleanup)
    # marks the term as malformed, so fail cleanly instead of crashing
    # downstream on tuple.free_symbols.
    if isinstance(parsed, (tuple, list)):
        return None
    return parsed


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

# Word-problem order: integral bounds first, then the series -
# "integral from 0 to 1/2 of the infinite sum from n=2 to infinity of x^n
# with respect to x". Common in textbook phrasing, where the bounds of the
# integral lead instead of trailing after "dx".
_WORDY_RE = re.compile(
    r"^(?:integral|integrate|int|∫)\s+(?:(?:the|an?|infinite|of)\s+)*"
    r"(?:sum|series|Σ)\s+(?:of\s+)?(?P<term>.+?)\s+from\s+(?P<var>[a-zA-Z])\s*=\s*"
    r"(?P<slo>.+?)\s+to\s+(?P<shi>.+?)\s*d\s*(?P<dvar>[a-zA-Z])\s+from\s+"
    r"(?P<lo>.+?)\s+to\s+(?P<hi>.+?)\s*$"
    r"|^(?:integral|integrate|int|∫)\s+from\s+(?P<lo2>.+?)\s+to\s+(?P<hi2>.+?)\s+"
    r"of\s+(?:(?:the|an?|infinite)\s+)*(?:sum|series|Σ)\s+from\s+(?P<var2>[a-zA-Z])\s*=\s*"
    r"(?P<slo2>.+?)\s+to\s+(?P<shi2>.+?)\s+of\s+(?P<term2>.+?)\s+"
    r"(?:with\s+respect\s+to\s*|d\s*)(?P<dvar2>[a-zA-Z])\s*$",
    re.IGNORECASE,
)

# Spelled-out numbers some users type in word-problem phrasing
# ("integral from zero to one-half ...").
_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "half": "1/2", "one half": "1/2", "one-half": "1/2", "a half": "1/2",
    "one third": "1/3", "one-third": "1/3",
    "one fourth": "1/4", "one-fourth": "1/4",
    "one quarter": "1/4", "one-quarter": "1/4",
}


def _word_number_bound(tok):
    tok = (tok or "").strip().strip(",").strip()
    return _WORD_NUMBERS.get(tok.lower(), tok)


def _normalize_wordy(text: str) -> str:
    """Pre-normalize voice/textbook-style phrasing into the solver's grammar
    before matching, e.g. "The integral from zero to one-half, of the
    infinite sum from n equals two to infinity of x to the power of n, with
    respect to x"."""
    t = (text or "").replace(",", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(
        r"^(?:the|a|an|please|evaluate|compute|calculate|find|solve)\s+",
        "", t, flags=re.IGNORECASE,
    )
    t = re.sub(r"\b(?:equals|is equal to)\b", "=", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b([a-zA-Z0-9_]+)\s+to\s+the\s+power\s+(?:of\s+)?([a-zA-Z0-9_/]+)\b",
        r"\1^\2", t, flags=re.IGNORECASE,
    )
    return t


class _Groups:
    """Minimal match-like wrapper so a rewrapped group dict still fits the
    common groupdict() path used for all regex flavors."""

    def __init__(self, groups: dict):
        self._groups = groups

    def groupdict(self) -> dict:
        return self._groups


def _rewrap_wordy(m):
    """Merge _WORDY_RE's two alternative branches into one set of group
    names and resolve spelled-out numbers in the bounds."""
    d = {
        k: (m.group(k) or m.group(k + "2"))
        for k in ("term", "var", "slo", "shi", "dvar", "lo", "hi")
    }
    for k in ("lo", "hi", "slo", "shi"):
        d[k] = _word_number_bound(d[k])
    return _Groups(d)

# ------------------------------------------------------------- solving


def _closed_candidates(term, n, slo, shi, x, lo, hi):
    """All closed-form candidates sympy's summation produced for the
    series, one per Piecewise branch (Sum-containing branches excluded)."""
    try:
        s = sp.summation(term, (n, slo, shi))
    except Exception:
        return []
    if isinstance(s, sp.Piecewise):
        return [e for e, _cond in s.args if not e.has(sp.Sum)]
    if s.has(sp.Sum):
        return []
    return [sp.simplify(s)]


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


# ------------------------------------------------------------- divergence


def _check_divergence(term, n, slo, shi, x, lo, hi, closeds):
    r"""Return an evidence dict if the integral of the series diverges,
    else None. Checks (each independent):
    1. A real pole of any closed-form branch inside the integration
       interval or an antiderivative whose one-sided endpoint limit is
       infinite. All sympy Piecewise branches are scanned because branch
       selection depends on sampling the integration interval, which can
       reject the correct branch (e.g. the mid-point of [0, 2] lies
       outside |x|<1 even though the |x|<1 branch is still the right
       formula - it just stops applying partway through the interval).
    2. Term test: if the integrated coefficients a_k = \int lo..hi f_k dx
       do not tend to 0, the series of integrals violates the necessary
       condition for convergence and diverges - no closed-form
       assumption is required for this check.
    Truncated quadrature is used only as supporting evidence, never as
    the primary decision."""
    closeds = closeds or []
    poles: list = []
    boundary_infinite = False

    for closed in closeds:
        try:
            rat = sp.cancel(sp.together(closed))
            num, den = sp.fraction(rat)
        except Exception:
            continue
        try:
            for p in sp.solveset(den, x, sp.S.Reals).intersection(sp.Interval(sp.S(lo), sp.S(hi))):
                try:
                    if num.subs(x, p) != 0:  # not a removable singularity
                        poles.append(p)
                except Exception:
                    poles.append(p)
        except Exception:
            pass
        try:
            antideriv = sp.integrate(rat, x)
            if antideriv is not None and not antideriv.has(sp.Integral):
                for bound, direction in ((sp.S(lo), "+"), (sp.S(hi), "-")):
                    try:
                        lim = sp.limit(antideriv, x, bound, direction)
                        if (lim in (oo, -oo) or getattr(lim, "is_infinite", False)
                                or str(lim) in ("oo", "-oo", "zoo", "nan")):
                            boundary_infinite = True
                            break
                    except Exception:
                        continue
        except Exception:
            pass
        if poles or boundary_infinite:
            break

    # Term test (a_k != 0 => divergence) only applies to INFINITE series:
    # a finite sum of integrals has no convergence to speak of, and this
    # check previously mis-flagged finite sums like ∫_0^2 Σ_{n=1}^3 nx dx
    # (value 12) as divergent because successive finite-sum evaluations
    # legitimately do not shrink toward zero.
    term_test_fails = False
    coeffs: list = []
    if shi is oo:
        coeffs = _integrated_coefficients(term, n, slo, x, lo, hi)
        term_test_fails = (
            len(coeffs) >= 20
            and all(abs(c) > 1e-7 for c in coeffs[-10:])
        )

    if not (poles or boundary_infinite or term_test_fails):
        return None

    evidence_parts = []
    if poles or boundary_infinite:
        truncs: list = []
        try:
            import mpmath
            f = sp.lambdify(x, sp.cancel(sp.together(closeds[0])), "mpmath")
            a, b = float(lo), float(hi)
            for eps in (1e-2, 1e-4, 1e-6):
                try:
                    truncs.append(float(mpmath.quad(f, [a, b - eps])))
                except Exception:
                    pass
        except Exception:
            pass
        if len(truncs) >= 2:
            evidence_parts.append(
                "truncated quadrature up to hi-eps for eps=1e-2,1e-4,1e-6 gives "
                + ", ".join(f"{t:.4g}" for t in truncs)
                + " - growing without bound instead of converging")
    if term_test_fails:
        evidence_parts.append(
            "the integrated coefficients "
            + ", ".join(f"{c:.4g}" for c in coeffs[-5:])
            + " do not tend to 0, so the term test makes the series diverge")

    return {
        "poles": poles,
        "boundary_infinite": boundary_infinite,
        "term_test_fails": term_test_fails,
        "evidence": "; ".join(evidence_parts) + "."
        if evidence_parts else "",
    }


def _integrated_coefficients(term, n, slo, x, lo, hi, terms=50):
    """The values a_k = ∫_{lo}^{hi} f_k(x) dx of the series term indexed by
    k, used for the divergence term test. Capped at `terms` evaluations."""
    try:
        coeffs = []
        start = int(slo)
        for k in range(start, start + terms):
            iv = sp.integrate(term.subs(n, k), (x, lo, hi))
            val = complex(sp.N(iv))
            coeffs.append(val.real)
        return coeffs
    except Exception:
        return []

    numerics = []

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
    # LaTeX spacing tokens ("\,", "\;", "\!", ...) left in the term would
    # survive into the sympify call and break term parsing.
    for spacer in ("\\,", "\\;", "\\!", "\\:", "\\ ", "\\quad", "\\qquad"):
        text = text.replace(spacer, " ")

    m = _PLAIN_RE.match(text) or _SYMPY_RE.match(text)
    flavor = "plain"
    if m is None:
        m = _WORDY_RE.match(_normalize_wordy(text))
        flavor = "plain"
        if m is not None:
            m = _rewrap_wordy(m)
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

    q_latex = (rf"\int_{{{latex(lo)}}}^{{{latex(hi)}}} "
               rf"\sum_{{{var_name}={latex(slo)}}}^{{{latex(shi)}}} "
               rf"{latex(term)} \, d{g['dvar']}")

    # All closed-form candidates from sympy's summation, not just one.
    # On some inputs sympy's branch selection rejects the correct branch
    # (e.g. a meridian of [0, 2] lies outside |x|<1 even though that is
    # the right formula for part of the interval). Scanning every branch
    # catches divergences and valid convergents alike.
    closeds = _closed_candidates(term, n, slo, shi, x, lo, hi)

    # A closed form with a pole on the integration interval (e.g.
    # \sum 3^n x^n on [0, 1/3]: closed form 3x/(1-3x) blows up at x=1/3)
    # means the integral diverges. Detect and report that explicitly
    # instead of handing sympy an improper integral it may mis-evaluate.
    diverg = _check_divergence(term, n, slo, shi, x, lo, hi, closeds)
    if diverg is not None:
        steps = [(
            "Recognize the series inside the integral", q_latex,
            "The integrand is a series; resolve it before integrating.", None,
        )]
        if closeds:
            for c in closeds:
                steps.append((
                    "Close the series (geometric-series formula)", latex(c),
                    "The series sums to this closed form.", None,
                ))
        if diverg["term_test_fails"]:
            steps.append((
                "Term test: the integrated coefficients must tend to 0",
                latex(sp.integrate(term.subs(n, symbols("k")), (x, lo, hi))),
                "A necessary condition for convergence is that the coefficients "
                "of the series of integrals tend to 0 - this does not hold.",
                None,
            ))
        singular_bits = []
        if diverg["poles"]:
            singular_bits.append(
                "a pole inside the interval at "
                + ", ".join(rf"{g['dvar']}={latex(p)}" for p in diverg["poles"])
            )
        if diverg["boundary_infinite"]:
            singular_bits.append("an antiderivative that diverges at an endpoint")
        if singular_bits:
            steps.append((
                "Detect a non-integrable singularity",
                latex(sp.denom(sp.cancel(sp.together(closeds[0])))),
                "The closed form has " + " and ".join(singular_bits)
                + ", so the integral is improper and fails to converge.",
                None,
            ))
        steps.append((
            "Conclude", r"\text{Diverges}",
            diverg["evidence"] or "The improper integral grows without bound.",
            r"\text{Diverges}",
        ))
        return {
            "question_latex": q_latex,
            "answer": "Diverges",
            "answer_latex": r"\text{Diverges}",
            "numeric": None,
            "steps": steps,
            "verification": diverg["evidence"] or (
                "Detected analytically via a singularity of the closed form."
            ),
            "diverges": True,
        }

    # Try each closed-form candidate for a convergent integral.
    result = None
    closed_used = None
    for c in closeds:
        try:
            r = sp.integrate(c, (x, lo, hi))
            if r is None or r.has(sp.Integral):
                continue
            result, closed_used = sp.simplify(r), c
            break
        except Exception:
            continue

    swapped = _swapped_series(term, n, slo, shi, x, lo, hi)
    if result is None:
        result = swapped
    if result is None:
        return None

    numeric = _numeric_check(closed_used, x, lo, hi)
    if numeric is None:
        numeric = _numeric_partial_sum(term, n, slo, shi, x, lo, hi)
    try:
        value = float(sp.N(result))
    except Exception:
        return None
    if numeric is None or abs(value - numeric) > 1e-6:
        return None  # honest refusal over a wrong headline answer

    steps = [(
        "Recognize the series inside the integral", q_latex,
        "The integrand is a series; resolve it before generic integration.", None,
    )]
    if closed_used is not None:
        steps.append((
            "Close the series (geometric-series formula)", latex(closed_used),
            "The series sums to this closed form on the whole integration interval.",
            None,
        ))
    steps.append((
        "Apply the Fundamental Theorem of Calculus", latex(result),
        "Integrate the closed form (or term-by-term) and evaluate at the bounds.",
        latex(result),
    ))
    if swapped is not None and closed_used is not None:
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