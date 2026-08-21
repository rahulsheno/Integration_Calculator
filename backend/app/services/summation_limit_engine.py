"""Solver for limits of partial sums: lim_{n->oo} Σ_{i=a}^{n} f(i, n).

No closed-form lookups or canned answers: the result is always derived
from general symbolic/numeric machinery:

1. Riemann-sum recognition. If the sum has the shape
   Σ_{i≈1}^{n} (1/n)·g(i/n) - detected by requiring n·f(n·t, n) to lose
   all dependence on n - then by the definition of the definite integral
   the limit equals ∫_0^1 g(t) dt, which is evaluated symbolically.
2. Closed-form partial sums. If sympy can find a closed form for the
   finite sum (telescoping, power sums, ...), the n→∞ limit is taken
   directly.

Both candidate answers are cross-checked against direct numeric partial
sums at increasing n (with the standard 1/n-error cancellation
extrapolation); a candidate that fails the cross-check is refused.
"""

import re

import sympy as sp
from sympy import latex, oo, symbols

_OO_TOKENS = {"oo", "inf", "infinity", "∞", "infty", "\\infty", "+oo", "+inf"}


# ------------------------------------------------------------- parsing

def _parse_bound(tok: str):
    tok = (tok or "").strip().strip("{}").replace("\\infty", "oo")
    if tok.lower() in _OO_TOKENS:
        return oo
    try:
        return sp.sympify(tok.replace("^", "**"))
    except Exception:
        return None


def _parse_term(tok: str):
    candidate = (tok or "").strip().strip("{}")
    if not candidate:
        return None
    candidate = candidate.replace("^", "**")
    parsed = None
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
    if isinstance(parsed, (tuple, list)):
        return None
    return parsed


def _strip_matching_braces(tok: str) -> str:
    """Unwrap a single pair of enclosing braces ONLY when they actually
    match - a blind strip('{}') destroys expressions like '\\frac{a}{b}'
    (the trailing brace gets removed, the leading backslash prevents the
    other), breaking every downstream balanced-brace rewrite."""
    while len(tok) > 1 and tok[0] == "{" and tok[-1] == "}":
        depth = 0
        close_pos = None
        for i, ch in enumerate(tok):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos == len(tok) - 1:
            tok = tok[1:-1].strip()
        else:
            break
    return tok


def _parse_latex_term(tok: str):
    tok = _strip_matching_braces((tok or "").strip())
    try:
        from latex2sympy2 import latex2sympy
        return latex2sympy(tok)
    except Exception:
        pass
    plain = (tok.replace("\\infty", "oo").replace("\\pi", "pi")
             .replace("\\cdot", "*").replace("\\left", "").replace("\\right", ""))
    for _ in range(20):
        replaced = _rewrite_balanced_latex_command(plain, r"\frac", 2, lambda a: f"(({a[0]})/({a[1]}))")
        replaced = _rewrite_balanced_latex_command(replaced, r"\sqrt", 1, lambda a: f"sqrt({a[0]})")
        if replaced == plain:
            break
        plain = replaced
    plain = (plain.replace("^{", "**(").replace("_{", "(").replace("}", ")")
             .replace("^", "**").replace("\\", ""))
    return _parse_term(plain)


def _rewrite_balanced_latex_command(text: str, command: str, arg_count: int, render) -> str:
    """Like math_ocr._replace_balanced_command: replace \\frac{a}{b} etc.
    correctly even when the arguments themselves contain nested braces
    (e.g. \\frac{n}{n^{2}+i^{2}}), which a brace-excluding regex cannot
    match - the very shape OCR models emit for rational terms."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(command, i):
            j = i + len(command)
            args = []
            ok = True
            for _ in range(arg_count):
                if j < len(text) and text[j] == "{":
                    depth, k = 0, j
                    while k < len(text):
                        if text[k] == "{":
                            depth += 1
                        elif text[k] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        k += 1
                    if depth != 0:
                        ok = False
                        break
                    args.append(text[j + 1:k])
                    j = k + 1
                else:
                    ok = False
                    break
            if ok:
                out.append(render(args))
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


# "limit of sum n/(n^2+i^2) from i=1 to n as n -> oo"
_PLAIN_RE = re.compile(
    r"^(?:limit|lim)\s+(?:of\s+)?(?:the\s+)?"
    r"(?:sum|series|Σ)\s+(?:of\s+)?(?P<term>.+?)\s+from\s+(?P<svar>[a-zA-Z])\s*=\s*"
    r"(?P<slo>\S+?)\s+to\s+(?P<shi>\S+?)\s+"
    r"(?:as|when)\s+(?P<nvar>[a-zA-Z])\s*->\s*(?P<nlo>\S+?)\s*$",
    re.IGNORECASE,
)

# "\lim_{n\to\infty} \sum_{i=1}^{n} \frac{n}{n^2+i^2}"
# Also accepts the "\operatorname*{lim}" spelling some OCR models emit -
# that variant is normalized to "\lim" in try_series_limit before matching.
_LATEX_RE = re.compile(
    r"^\s*\\lim\s*_"
    r"\{\s*(?P<nvar>[a-zA-Z])\s*\\(?:to|rightarrow)\s*\\infty\s*\}\s*"
    r"\\sum\s*_\{\s*(?P<svar>[a-zA-Z])\s*=\s*(?P<slo>[^{}]+?)\s*\}\s*"
    r"\^\{(?P<shi>[^{}]+?)\}\s*(?P<term>.+?)\s*$",
    re.IGNORECASE,
)


# ------------------------------------------------------------- solving

def _numeric_partial_sums(term, svar, n_var, slo, shi):
    """Direct numeric partial sums at growing n, extrapolated to n=oo.

    The endpoint error of sums like Σ_{i=1}^n (1/n)g(i/n) is O(1/n)
    (Euler-Maclaurin: S(n) = I + c₁/n + c₂/n² + ...), so evaluating at a
    few growing n and fitting I + c₁/n gives the limit far more tightly
    than extrapolated-from-2-points Richardson (which assumed O(n⁻²) and
    produced the wrong constant)."""
    if not (slo.is_number and shi.has(n_var)):
        return None, None
    try:
        import mpmath
        mpmath.mp.dps = 15
        f = sp.lambdify([n_var, svar], term, "mpmath")

        def partial(N) -> float:
            start = int(sp.N(slo))
            end = int(sp.N(shi.subs(n_var, N)))
            if end - start + 1 > 200000:
                return None
            total = mpmath.mpf(0)
            for k in range(start, end + 1):
                v = f(N, k)
                if isinstance(v, complex) or v != v:
                    return None
                total += v
            return float(total)

        Ns = (1000, 2000, 4000)
        vals = []
        for N in Ns:
            v = partial(N)
            if v is None:
                return None, None
            vals.append(v)
    except Exception:
        return None, None

    # Fit I + a*(1/n) through the 3 points exactly (linear system in I, a).
    try:
        import numpy as np
        xs = np.array([1.0 / N for N in Ns])
        ys = np.array(vals)
        coef, *_ = np.linalg.lstsq(np.vstack([np.ones_like(xs), xs]).T, ys, rcond=None)
        extrapolated = float(coef[0])
    except Exception:
        extrapolated = vals[-1]
    return (Ns, vals), extrapolated


def _try_riemann(term, svar, n_var, slo, shi):
    """Return g(t) if Σ f(i,n) is a Riemann sum on [0,1] of g, else None.
    Condition: f(n·t, n)·n must be independent of n after simplification.
    Requires the upper bound to grow exactly like n (so the partition width
    1/n covers [0,1]) and the lower bound to be a finite constant."""
    if not (slo.is_number and shi.has(n_var)):
        return None
    try:
        if not (sp.Poly(shi, n_var).degree() == 1 and
                sp.Poly(shi, n_var).LC() == 1):
            return None
    except Exception:
        return None
    t = sp.Symbol("t")
    try:
        g = sp.cancel(sp.together(sp.expand(term.subs(svar, n_var * t) * n_var)))
        g = sp.simplify(g)
        if g.has(n_var) or g.has(svar):
            return None
        return g
    except Exception:
        return None


def _try_closed_partial(term, svar, n_var, slo, shi):
    """Return the n->oo limit if sympy can close-form the finite sum."""
    try:
        s = sp.summation(term, (svar, slo, shi))
        if s.has(sp.Sum) or s.has(sp.Integral):
            s = sp.simplify(s.doit())
        if s.has(sp.Sum) or s.has(sp.Integral):
            return None, None
        s = sp.simplify(s)
        lim = sp.limit(s, n_var, oo)
        if lim.has(sp.Limit) or lim.has(sp.Sum):
            return None, None
        return s, sp.simplify(lim)
    except Exception:
        return None, None


# ------------------------------------------------------------- entry

def try_series_limit(expression: str):
    """Return a result dict if `expression` is a limit of a partial sum
    (as n -> infinity), else None (caller falls through)."""
    if not expression:
        return None
    low = expression.lower()
    if "lim" not in low:
        return None
    if "sum" not in low and "Σ" not in expression and "\\sum" not in expression:
        return None

    text = expression.strip().strip("$")
    for spacer in ("\\,", "\\;", "\\!", "\\:", "\\ ", "\\quad", "\\qquad"):
        text = text.replace(spacer, " ")
    # OCR engines sometimes emit "\operatorname*{lim}" instead of "\lim";
    # normalize before the single \lim pattern has to handle every guess.
    text = text.replace("\\operatorname*{\\lim}", "\\lim")
    text = text.replace(r"\operatorname*{lim}", r"\lim")

    m = _PLAIN_RE.match(text)
    flavor = "plain"
    if m is None:
        m = _LATEX_RE.match(text)
        flavor = "latex"
    if m is None:
        return None
    g = m.groupdict()

    term = (_parse_latex_term if flavor == "latex" else _parse_term)(g["term"])
    if term is None:
        return None

    svar_name, nvar_name = g["svar"], g["nvar"]
    svar, n_var = symbols(svar_name), symbols(nvar_name)
    slo, shi = _parse_bound(g["slo"]), _parse_bound(g["shi"])
    n_target = _parse_bound(g["nlo"]) if g.get("nlo") else oo
    if slo is None or shi is None or n_target != oo:
        return None
    if not term.free_symbols <= {svar, n_var}:
        return None

    samples, extrapolated = _numeric_partial_sums(term, svar, n_var, slo, shi)

    riemann_g = _try_riemann(term, svar, n_var, slo, shi)
    closed_partial, closed_limit = _try_closed_partial(term, svar, n_var, slo, shi)

    result = None
    used = None
    if closed_limit is not None:
        result, used = closed_limit, "closed"
    elif riemann_g is not None:
        try:
            r = sp.integrate(riemann_g, (sp.Symbol("t"), 0, 1))
            if r is not None and not r.has(sp.Integral):
                result, used = sp.simplify(r), "riemann"
        except Exception:
            result = None

    if result is None:
        return None
    try:
        value = float(sp.N(result))
    except Exception:
        return None
    if extrapolated is None or abs(value - extrapolated) > 1e-5:
        return None  # honest refusal over a wrong headline answer
    if closed_limit is not None and riemann_g is not None:
        try:
            r2 = sp.integrate(riemann_g, (sp.Symbol("t"), 0, 1))
            if abs(float(sp.N(r2)) - value) > 1e-5:
                return None  # two methods disagree -> refuse, never guess
        except Exception:
            pass

    q_latex = (rf"\lim_{{{nvar_name} \to \infty}} "
               rf"\sum_{{{svar_name}={latex(slo)}}}^{{{latex(shi)}}} {latex(term)}")

    steps = [(
        "Recognize a limit of partial sums", q_latex,
        "The quantity under the limit is a finite sum whose number of terms "
        f"grows with {nvar_name}.", None,
    )]

    if riemann_g is not None:
        steps.append((
            "Rewrite as a Riemann sum",
            rf"S_{{n}} = \sum_{{{svar_name}={latex(slo)}}}^{{{latex(shi)}}} "
            rf"\frac{{1}}{{{nvar_name}}} "
            rf"g\!\left(\frac{{{svar_name}}}{{{nvar_name}}}\right),\qquad "
            rf"g(t) = {latex(riemann_g)}",
            f"Factoring gives each term as (1/{nvar_name})·g({svar_name}/{nvar_name}); "
            "this is a Riemann sum on [0, 1] with partition width "
            rf"1/{nvar_name}.", None,
        ))
        steps.append((
            "Pass to the definite integral (definition of the integral)",
            rf"\lim_{{n \to \infty}} S_n = \int_0^1 {latex(riemann_g)} \, dt",
            "By the definition of the Riemann integral, a smooth integrand's "
            "Riemann sums converge to the integral.", None,
        ))

    if closed_partial is not None:
        steps.append((
            "Close the partial sum symbolically",
            rf"S_n = {latex(closed_partial)}",
            "The finite sum evaluates to a closed form in terms of "
            f"{nvar_name}.", None,
        ))
        steps.append((
            "Take the limit of the partial sum",
            latex(result),
            f"Letting {nvar_name} → ∞ in the closed form gives the limit.",
            latex(result),
        ))
    else:
        steps.append((
            "Evaluate the integral",
            latex(result),
            "Compute the definite integral via the Fundamental Theorem of Calculus.",
            latex(result),
        ))

    ns, vals = samples
    return {
        "question_latex": q_latex,
        "answer": str(result),
        "answer_latex": latex(result),
        "numeric": value,
        "steps": steps,
        "verification": (
            "Verified numerically: direct partial sums at "
            + ", ".join(f"n={N}" for N in ns) + " give "
            + ", ".join(f"{v:.10g}" for v in vals)
            + f"; error-canceled extrapolation = {extrapolated:.10g} agrees "
            + f"with the symbolic value {value:.10g}."
        ),
    }
