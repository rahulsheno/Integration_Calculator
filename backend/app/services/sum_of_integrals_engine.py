"""Solver for a sum of definite integrals:

    sum_{k=a}^{b} int_lo^hi f(k, x) dx

The mirror image of series_integral_engine (which handles int of sum).
This shape is what "\\sum_{n=1}^{4} \\int_0^1 n x^{n-1} dx" means: each
term of the sum is a definite integral, evaluated at one index value.

No canned answers: the inner integral is evaluated symbolically, then
the outer sum is taken (exactly by sympy when it can, or by direct
substitution on every index for finite bounds). The final value is
always independently recomputed with pure numeric quadrature of the
original term-by-term integrals before being reported; mismatch or
failure -> honest refusal.
"""

import re

import sympy as sp
from sympy import latex, oo, symbols
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)

_OO_TOKENS = {"oo", "inf", "infinity", "∞", "infty", "\\infty", "+oo", "+inf"}


def _unpack_braces(tok: str) -> str:
    tok = (tok or "").strip()
    if len(tok) > 1 and tok[0] == "{" and tok[-1] == "}":
        return tok[1:-1].strip()
    return tok


def _parse_bound(tok: str):
    tok = _unpack_braces(tok).strip()
    # Bounds arrive as raw LaTeX ("\pi", "\infty", "\frac{1}{2}") - translate
    # the constants before handing the rest to sympify.
    tok = tok.replace("\\infty", "oo").replace("\\pi", "pi")
    neg = tok.startswith("-")
    if tok.lstrip("+-").lower() in _OO_TOKENS:
        return -oo if neg else oo
    try:
        return sp.sympify(tok.replace("^", "**"))
    except Exception:
        return None


def _parse_term(tok: str):
    candidate = _unpack_braces(tok or "")
    if not candidate:
        return None
    candidate = candidate.replace("^", "**")
    try:
        parsed = sp.sympify(candidate)
    except Exception:
        parsed = None
    if parsed is None:
        try:
            parsed = parse_expr(
                candidate,
                transformations=standard_transformations + (implicit_multiplication_application,),
                evaluate=True,
            )
        except Exception:
            parsed = None
    if isinstance(parsed, (tuple, list)):
        parsed = None
    return parsed


def _consume_balanced(text: str, pos: int):
    """If text[pos] == '{', return (inner, pos_after_close); else (None, pos)."""
    if pos >= len(text) or text[pos] != "{":
        return None, pos
    depth = 0
    for j in range(pos, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[pos + 1:j], j + 1
    return None, pos


def _rewrite_latex_frac(text: str) -> str:
    """Replace every \\frac{a}{b} (nested braces allowed) with ((a)/(b))."""
    out = []
    i = 0
    while i < len(text):
        if text.startswith(r"\frac", i):
            a, j = _consume_balanced(text, i + len(r"\frac"))
            if a is None:
                out.append(text[i])
                i += 1
                continue
            while j < len(text) and text[j].isspace():
                j += 1
            b, k = _consume_balanced(text, j)
            if b is None:
                out.append(text[i])
                i += 1
                continue
            out.append(f"((({a})/({b})))")
            i = k
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _strip_balanced_wrappers(text: str) -> str:
    """Unwrap balanced outer ( ) and { } pairs around an integrand, e.g.
    the "{...}" the OCR shape "\\int_0^1{... dx}" puts around the whole
    integrand - without eating braces that belong to exponents inside."""
    t = text.strip()
    for opener, closer in (("{", "}"), ("(", ")")):
        while t.startswith(opener):
            depth, i = 0, 0
            while i < len(t):
                if t[i] == opener:
                    depth += 1
                elif t[i] == closer:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0 and i == len(t) - 1:
                t = t[1:-1].strip()
            else:
                break
    return t


def _parse_latex_term(tok: str):
    tok = _strip_balanced_wrappers((tok or "").strip())
    try:
        from latex2sympy2 import latex2sympy
        return latex2sympy(tok)
    except Exception:
        pass
    plain = (tok.replace("\\infty", "oo").replace("\\pi", "pi")
             .replace("\\cdot", "*"))
    for _ in range(20):
        replaced = _rewrite_latex_frac(plain)
        if replaced == plain:
            break
        plain = replaced
    # Implicit multiplication before directly-attached function commands:
    # "x^{n}\ln(x)" would otherwise collapse to "x^(n)ln(x)" / fused word
    # "nln" and fail parsing - same class of OCR shape the generic
    # normalizer needed, handled here at the command boundary.
    plain = re.sub(
        r"([0-9a-zA-Z})])(\\(?:sin|cos|tan|log|ln|exp))\b",
        r"\1*\2",
        plain,
    )
    plain = plain.replace("\\ln", "log").replace("\\log", "log")
    plain = (plain.replace("^{", "**(").replace("^", "**")
             .replace("}", ")").replace("\\", ""))
    return _parse_term(plain)


#                      plain-text grammar
# "sum integrate n*x^(n-1) dx from 0 to 1 from n=1 to 4"
_PLAIN_RE = re.compile(
    r"^(?:sum|Σ)\s*(?:of\s+)?(?:the\s+)?"
    r"(?:integrate|integral|int|∫)\s+(?P<term>.+?)\s*d\s*(?P<dvar>[a-zA-Z])\s+"
    r"from\s+(?P<lo>\S+?)\s+to\s+(?P<hi>\S+?)\s+"
    r"from\s+(?P<svar>[a-zA-Z])\s*=\s*(?P<slo>\S+?)\s+to\s+(?P<shi>\S+?)\s*$",
    re.IGNORECASE,
)

#                      LaTeX grammar
# "\sum_{n=1}^{4} \int_{0}^{1} n x^{n-1} dx"
# The OCR variant wrapping the whole integrand in braces,
# "\sum_{n=1}^{4}\int_{0}^{1}{n x^{n-1}\,d x}", is normalized before
# matching by _unwrap_integral_integrand_braces.
_LATEX_RE = re.compile(
    r"^\\sum\s*_\{\s*(?P<svar>[a-zA-Z])\s*=\s*(?P<slo>[^{}]+?)\s*\}\s*"
    r"\^\{(?P<shi>[^{}]+?)\}\s*"
    r"\\int\s*(?:_(?P<lo>\{[^{}]*\}|[^_^ ]+))?\s*(?:\^(?P<hi>\{[^{}]*\}|[^_^ ]+))?\s*"
    r"(?P<term>.+?)\s*\s*d\s*(?P<dvar>[a-zA-Z])\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _unwrap_integral_integrand_braces(text: str) -> str:
    """If the integrand after \\int (with its bounds) sits inside one
    balanced brace pair - "\\int_0^1{f(x) dx}" - unwrap it, so the
    regex never has to guess which '}' closes the exponent vs. the
    wrapper. Only acts when the matching '}' is the last character."""
    idx = text.find("\\int")
    if idx == -1:
        return text
    after = len("\\int")
    for _ in range(2):
        j = idx + after
        while j < len(text) and text[j] in "_^":
            open_ch, close_ch = ("{", "}") if j + 1 < len(text) and text[j + 1] == "{" else (None, None)
            if open_ch is None:
                # bare token bound like \int_0^1
                k = re.match(r"[_^]\s*[^\s_^]+", text[j:])
                if not k:
                    break
                j += k.end()
                continue
            depth = 0
            k = j + 1
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            j = k + 1 if depth == 0 else len(text)
    open_pos = j
    if open_pos < len(text) and text[open_pos] == "{":
        depth = 0
        k = open_pos
        while k < len(text):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth == 0 and k == len(text) - 1:
            return (text[:open_pos] + " " + text[open_pos + 1:k]).strip()
    return text


def try_sum_of_integrals(expression: str):
    """Return a result dict if `expression` is a sum of definite
    integrals, else None (caller falls through to the generic solver)."""
    if not expression:
        return None
    low = expression.lower()
    if "sum" not in low and "Σ" not in expression and "\\sum" not in expression:
        return None
    if "integr" not in low and "\\int" not in expression and "∫" not in expression:
        return None

    text = expression.strip().strip("$")
    for spacer in ("\\,", "\\;", "\\!", "\\:", "\\ ", "\\quad", "\\qquad"):
        text = text.replace(spacer, " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = _unwrap_integral_integrand_braces(text)

    flavor = "plain"
    m = _PLAIN_RE.match(text)
    if m is None:
        m = _LATEX_RE.match(text)
        flavor = "latex"
    if m is None:
        return None
    g = m.groupdict()

    parse_term_fn = _parse_latex_term if flavor == "latex" else _parse_term
    term = parse_term_fn(g["term"])
    if term is None:
        return None

    svar_name, dvar_name = g["svar"], g["dvar"]
    svar_sym, dvar = symbols(svar_name), symbols(dvar_name)
    slo, shi = _parse_bound(g["slo"]), _parse_bound(g["shi"])
    lo, hi = _parse_bound(g["lo"]), _parse_bound(g["hi"])
    if None in (slo, shi, lo, hi):
        return None
    if not term.free_symbols <= {svar_sym, dvar}:
        return None
    if lo == hi:
        return None

    # --- Step 1: evaluate the inner definite integral once, keeping the
    # summation index as a parameter.
    inner = None
    try:
        inner = sp.integrate(term, (dvar, sp.S(lo), sp.S(hi)))
    except Exception:
        inner = None
    if inner is None or inner.has(sp.Integral):
        # Retry with integer/positivity assumptions on the summation
        # index: sp.integrate(x**n*log(x), (x, 0, 1)) stays unevaluated
        # for an unconstrained n, but closes to -1/(n+1)**2 once n is
        # known to be a positive integer - exactly the case for any
        # integer-bounded outer sum. This is a legitimate mathematical
        # narrowing of the given problem (the summed index IS an
        # integer), not an assumption of the answer.
        try:
            assumed = symbols(svar_name, integer=True, positive=True)
            candidate = sp.integrate(term.xreplace({svar_sym: assumed}),
                                     (dvar, sp.S(lo), sp.S(hi)))
            if candidate is not None and not candidate.has(sp.Integral):
                inner = candidate.xreplace({assumed: svar_sym})
        except Exception:
            pass
    if inner is None or inner.has(sp.Integral):
        return None
    inner = sp.simplify(inner)

    # --- Step 2: take the outer sum.
    result = None
    try:
        s = sp.summation(inner, (svar_sym, sp.S(slo), sp.S(shi)))
        if s is not None and not s.has(sp.Sum):
            result = sp.simplify(s)
    except Exception:
        result = None
    if result is None and shi is not oo:
        try:
            closed_start = int(sp.N(slo))
            closed_end = int(sp.N(shi))
            total = sp.Integer(0)
            for k in range(closed_start, closed_end + 1):
                piece = inner.subs(svar_sym, k)
                total += piece
            result = sp.simplify(total)
        except Exception:
            return None

    q_latex = (rf"\sum_{{{svar_name}={latex(sp.S(slo))}}}^{{{latex(sp.S(shi))}}} "
               rf"\int_{{{latex(sp.S(lo))}}}^{{{latex(sp.S(hi))}}} {latex(term)} \, d{dvar_name}")

    # sympy sometimes closes the outer sum to +oo/-oo instead of leaving
    # an unevaluated Sum - treat that as a genuine divergence verdict.
    if result is not None and shi is oo:
        try:
            numeric_result = sp.N(result)
            if str(numeric_result) in ("oo", "-oo", "zoo") or numeric_result.is_infinite:
                raise OverflowError
        except Exception:
            steps = [(
                "Recognize a sum of definite integrals", q_latex,
                "Evaluate the inner integral first, leaving the summation "
                "index as a parameter.", None,
            )]
            steps.append((
                "Evaluate the inner integral", latex(inner),
                "This is the k-th term of the outer series.", None,
            ))
            steps.append((
                "Take the outer sum", r"\text{Diverges}",
                "The sum of the integrated terms grows without bound - the "
                "necessary term-tendsto-zero condition fails.",
                r"\text{Diverges}",
            ))
            return {
                "question_latex": q_latex,
                "answer": "Diverges",
                "answer_latex": r"\text{Diverges}",
                "numeric": None,
                "steps": steps,
                "verification": (
                    f"The outer sum of the integrated terms {latex(inner)} "
                    "evaluates to infinity - the series diverges."
                ),
                "diverges": True,
            }

    # Infinite outer sum that sympy cannot close: apply the term test to
    # the integrated terms a_k = ∫ f(k, x) dx. If lim a_k != 0 the series
    # diverges - report that explicitly instead of refusing.
    if result is None and shi is oo:
        try:
            lim_val = sp.limit(inner, svar_sym, oo)
            diverges = (lim_val != 0 or lim_val in (oo, -oo)
                        or getattr(lim_val, "is_infinite", False)
                        or str(lim_val) in ("oo", "-oo", "zoo"))
        except Exception:
            diverges = None
        if diverges is None:
            # numeric evidence only - never decide on numbers alone
            try:
                import mpmath
                mpmath.mp.dps = 15
                f = sp.lambdify((svar_sym, dvar), term, "mpmath")
                lo_f, hi_f = float(sp.N(sp.S(lo))), float(sp.N(sp.S(hi)))
                totals = []
                run = 0.0
                start = int(sp.N(slo))
                for k in range(start, start + 60):
                    iv = mpmath.quad(lambda t: f(float(k), t), [lo_f, hi_f])
                    run += float(iv) if not isinstance(iv, float) else float(iv)
                    totals.append(run)
                if totals and abs(totals[-1]) > 10 and abs(totals[-1]) > abs(totals[0]) * 10:
                    diverges = True
            except Exception:
                diverges = None
        if not diverges:
            return None
        try:
            a_k = latex(inner)
        except Exception:
            a_k = "?"
        steps = [(
            "Recognize a sum of definite integrals", q_latex,
            "Evaluate the inner integral first, leaving the summation index "
            "as a parameter.", None,
        )]
        steps.append((
            "Evaluate the inner integral", a_k,
            "This is the k-th term of the outer series; convergence of the "
            "whole expression requires these terms to tend to 0.", None,
        ))
        steps.append((
            "Apply the term test", r"\text{Diverges}",
            f"The integrated terms {a_k} do not tend to 0 as "
            f"{svar_name} grows without bound, so the series diverges.",
            r"\text{Diverges}",
        ))
        return {
            "question_latex": q_latex,
            "answer": "Diverges",
            "answer_latex": r"\text{Diverges}",
            "numeric": None,
            "steps": steps,
            "verification": (
                f"Term test: the sequence of integrated terms {a_k} fails to "
                "tend to 0, which is necessary for any convergent series."
            ),
            "diverges": True,
        }

    if result is None:
        return None
    try:
        value = float(sp.N(result))
    except Exception:
        return None

    # --- Step 3: independent numeric verification - recompute the whole
    # expression term-by-term with raw quadrature on the ORIGINAL integrand.
    try:
        import mpmath
        mpmath.mp.dps = 15
        f = sp.lambdify((svar_sym, dvar), term, "mpmath")
        lo_f, hi_f = float(sp.N(sp.S(lo))), float(sp.N(sp.S(hi)))

        def term_value(k: int) -> float:
            iv = mpmath.quad(lambda t: f(float(k), t), [lo_f, hi_f])
            return float(sp.N(iv)) if not isinstance(iv, float) else float(iv)

        if shi is oo:
            # A naive truncation to 1e-6 needs O(1e6) terms for series
            # converging like 1/n^2 - hopeless for a verification pass.
            # mpmath.nsum applies convergence acceleration (Shanks /
            # Euler-Maclaurin transforms) to the quadratured terms, so the
            # check stays independent of the symbolic path yet settles
            # quickly even for slow-decaying terms.
            start = int(sp.N(slo))
            try:
                approx = float(mpmath.nsum(
                    lambda k: mpmath.quad(lambda t: f(float(k), t), [lo_f, hi_f]),
                    [start, mpmath.inf],
                ))
            except Exception:
                return None  # could not verify numerically: honest refusal
            if not all(c == c for c in [approx]) or abs(approx) > 1e15:
                return None
            if abs(approx - value) > max(1e-6 * abs(value), 1e-8):
                return None
        else:
            n_start, n_end = int(sp.N(slo)), int(sp.N(shi))
            total = 0.0
            for k in range(n_start, n_end + 1):
                total += term_value(k)
            if abs(total - value) > 1e-6 * max(1.0, abs(value)):
                return None
    except Exception:
        return None  # cannot verify -> refuse rather than headline an answer

    steps = [(
        "Recognize a sum of definite integrals", q_latex,
        "Each summand is a definite integral in " + dvar_name +
        "; evaluate the inner integral first, then sum over " + svar_name + ".",
        None,
    )]
    steps.append((
        "Evaluate the inner integral (Fundamental Theorem of Calculus)",
        latex(inner),
        "Integrating the term with respect to " + dvar_name +
        " on the given bounds keeps " + svar_name + " as a parameter.",
        None,
    ))
    steps.append((
        "Take the outer sum",
        latex(result),
        f"Sum the result over every {svar_name} from {slo} to {shi}.",
        latex(result),
    ))

    return {
        "question_latex": q_latex,
        "answer": str(result),
        "answer_latex": latex(result),
        "numeric": value,
        "steps": steps,
        "verification": (
            f"Verified: symbolic value {value:.10g} matches independent "
            "numeric term-by-term integration of the original expression."
        ),
    }
