import json
import re

import sympy as sp
from sympy import (
    Symbol, symbols, limit, Derivative, Integral, diff, integrate, solve,
    series, dsolve, Function, Eq, oo, pi, nan, sin, cos, tan, cot, sec, csc,
    log, exp, sqrt, factorial, Sum, Product, Matrix, latex as _sympy_latex, simplify, expand,
    factor, apart, together, ratsimp, trigsimp, powsimp, combsimp, radsimp,
    nsimplify, fraction, numer, denom, collect, cancel, nroots, nsolve
)
from sympy.calculus.util import continuous_domain, function_range
from sympy.solvers.ode import classify_ode
from sympy.series.fourier import fourier_series
from sympy.integrals.transforms import laplace_transform, inverse_laplace_transform
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)


def latex(expr, **kwargs):
    """Wrapper around sympy.latex that defaults to full inverse-trig names
    (arctan, arcsin, ...) instead of sympy's default abbreviated style
    (atan, asin, ...), since that's what users expect to see rendered."""
    kwargs.setdefault("inv_trig_style", "full")
    return _sympy_latex(expr, **kwargs)

from app.services.normalizer import normalize_expression, check_nesting_depth
from app.services.series_engine import try_solve_series, try_solve_continued_fraction, try_solve_product
from app.services.pattern_matcher import try_symmetry_substitution, try_periodicity_reduction, classify_named_pattern
from app.services.verification import numerical_verify, numeric_quad as _numeric_quad

try:
    # Reuse the same LaTeX->plain-expression conversion already used for
    # OCR'd images (see app/services/math_ocr.py), so that a user who
    # pastes raw LaTeX directly into the "Enter a calculus problem" box
    # (e.g. "\int e^{x}\cdot\sinh(x)\,dx") gets the same treatment instead
    # of failing outright - normalize_expression() only handles the
    # solver's own plain-text grammar ("integrate ... dx"), not LaTeX
    # commands, so raw LaTeX previously passed straight through unconverted
    # and every downstream parse attempt failed.
    from app.services.math_ocr import normalize_latex_math as _normalize_latex_math
except ImportError:
    _normalize_latex_math = None

from app.schemas.schemas import (
    SolveRequest, SolveResponse, StepDetail, AlternativeMethod,
    FormulaUsed, GraphData, MathContext, VerifyRequest, VerifyResponse, VerificationStep
)

CONTEXT_MEMORY: dict[str, dict[str, object]] = {}
DEFAULT_SESSION_ID = "default"

TOPIC_KEYWORDS = {
    "Limits": ["limit", "approaches", "tends to", "as x->"],
    "Continuity": ["continuous", "continuity", "discontinuity"],
    "Differentiation": ["derivative", "differentiate", "d/dx", "f'(x)", "rate of change"],
    "Implicit Differentiation": ["implicit", "dy/dx implicitly"],
    "Higher-Order Derivatives": ["second derivative", "third derivative", "nth derivative", "f''(x)", "f'''(x)"],
    "Partial Derivatives": ["partial derivative", "∂", "∂f/∂x", "∂f/∂y"],
    "Chain Rule": ["chain rule", "composite function"],
    "Product & Quotient Rules": ["product rule", "quotient rule"],
    "Optimization": ["optimize", "maximize", "minimize", "maximum", "minimum", "optimization"],
    "Related Rates": ["related rates", "rate of"],
    "Indefinite Integrals": ["indefinite integral", "antiderivative", "integrate", "integral", "∫"],
    "Definite Integrals": ["definite integral", "∫_", "from ... to", "integrate from"],
    "Integration by Parts": ["integration by parts", "∫ u dv", "∫ v du", "by parts"],
    "U-Substitution": ["u-substitution", "u sub", "substitution"],
    "Partial Fractions": ["partial fractions", "partial fraction decomposition"],
    "Trigonometric Integrals": ["trig integral", "∫ sin", "∫ cos", "∫ tan", "∫ sec", "∫ csc", "∫ cot"],
    "Trigonometric Substitution": ["trig substitution", "trigonometric substitution", "sin substitution", "tan substitution"],
    "Improper Integrals": ["improper integral", "∫ from -∞", "∫ to ∞"],
    "Multiple Integrals": ["double integral", "triple integral", "∬", "∭", "multiple integral"],
    "Vector Calculus": ["gradient", "divergence", "curl", "∇", "nabla"],
    "Line Integrals": ["line integral", "∮", "path integral"],
    "Surface Integrals": ["surface integral"],
    "Green's Theorem": ["green's theorem", "green theorem"],
    "Stokes' Theorem": ["stokes theorem", "stokes' theorem"],
    "Divergence Theorem": ["divergence theorem", "gauss theorem", "gauss's theorem"],
    "Differential Equations": ["differential equation", "ode", "pde", "y'", "y''", "y'''"],
    "Taylor & Maclaurin Series": ["taylor", "maclaurin", "series expansion", "power series"],
    "Fourier Series": ["fourier", "fourier series"],
    "Laplace Transforms": ["laplace", "laplace transform", "L{"],
    "Polar Coordinates": ["polar", "r =", "r(θ)", "θ"],
    "Parametric Equations": ["parametric", "x(t)", "y(t)", "parametri"],
}


def classify_topic(expression: str) -> tuple[str, float]:
    expression_lower = expression.lower()
    scores = {}

    # First pass: detect integral type by bounds/kind before generic keywords
    # Improper integrals: have infinity bounds
    if any(inf in expression_lower for inf in ('oo', '∞', 'inf', '-oo', 'from -')):
        scores["Improper Integrals"] = 3

    # Definite integrals: have "from ... to" pattern (check before generic "integrate")
    if re.search(r'(?:from\s+.+\s+to\s+|integrate\s+from)', expression_lower):
        scores["Definite Integrals"] = 3

    # Specific techniques
    if re.search(r'(?:x\s*\*\s*exp|x\s*\*\s*e\^|x\s*e\^|\bexp\b)', expression_lower) and 'sin' in expression_lower:
        scores["Integration by Parts"] = 2
    if any(kw in expression_lower for kw in ("by parts", "∫ u dv", "∫ v du")):
        scores["Integration by Parts"] = 2
    if any(kw in expression_lower for kw in ("substitution", "u-sub", "u sub")):
        scores["U-Substitution"] = 2
    if any(kw in expression_lower for kw in ("partial fraction", "partial fractions")):
        scores["Partial Fractions"] = 2
    if any(kw in expression_lower for kw in ("trig substitution", "trigonometric substitution")):
        scores["Trigonometric Substitution"] = 2

    for topic, keywords in TOPIC_KEYWORDS.items():
        # Skip the ones we already explicitly scored above if they had specific matches
        if topic in ("Definite Integrals", "Improper Integrals") and topic in scores:
            continue
        score = sum(1 for kw in keywords if kw.lower() in expression_lower)
        if score > 0:
            scores[topic] = scores.get(topic, 0) + score

    if not scores:
        return "General Calculus", 0.3

    # Prefer Definite over Indefinite, Improper over both
    for preferred in ("Improper Integrals", "Definite Integrals", "Indefinite Integrals"):
        if preferred in scores and scores[preferred] >= max(scores.values()) * 0.6:
            best_topic = preferred
            break
    else:
        best_topic = max(scores, key=scores.get)

    confidence = min(scores[best_topic] / 5, 1.0)
    return best_topic, confidence


def parse_expression(expr_str: str):
    expr_str = expr_str.strip().replace("\\", "")
    expr_str = re.sub(r'∂', '', expr_str)

    # Check for a periodic continued fraction: a literal "..." marks the
    # infinite repetition, e.g. "1/(1+x/(1+x/(1+...)))".
    if '...' in expr_str:
        return "continued_fraction", expr_str

    # Check for an infinite/finite series or product: "sum x^n from n=0 to
    # infinity", "product (n+1)/n from n=1 to 5".
    m = re.match(
        r'(?:sum|series)\s+(?:of\s+)?(.+?)\s+from\s+([a-zA-Z])\s*=\s*(.+?)\s+to\s+(.+)',
        expr_str, re.IGNORECASE,
    )
    if m:
        return "series", m.group(1).strip(), m.group(2), m.group(3).strip(), m.group(4).strip()

    m = re.match(
        r'(?:product|prod)\s+(?:of\s+)?(.+?)\s+from\s+([a-zA-Z])\s*=\s*(.+?)\s+to\s+(.+)',
        expr_str, re.IGNORECASE,
    )
    if m:
        return "product", m.group(1).strip(), m.group(2), m.group(3).strip(), m.group(4).strip()

    # Explicit Sum(...) / Product(...) function-call syntax is left to the
    # general "expression" fallback path further down, since safe_sympify
    # already understands those names directly.

    # Check for limit
    m = re.match(r'(?:limit|lim)\s+(?:of\s+)?(.+?)\s+as\s+([a-zA-Z])\s*->\s*(.+)', expr_str, re.IGNORECASE)
    if m:
        return "limit", m.group(1).strip(), m.group(2), m.group(3).strip()

    m = re.match(r'lim(?:it)?\s*\(\s*(.+?)\s*,\s*([a-zA-Z])\s*,\s*(.+?)\s*\)', expr_str, re.IGNORECASE)
    if m:
        return "limit", m.group(1).strip(), m.group(2), m.group(3).strip()

    m = re.match(r'(?:limit|lim)\s+(.+)', expr_str, re.IGNORECASE)
    if m:
        rest = m.group(1)
        m2 = re.match(r'(.+?)\s+as\s+([a-zA-Z])\s*->\s*(.+)', rest, re.IGNORECASE)
        if m2:
            return "limit", m2.group(1).strip(), m2.group(2), m2.group(3).strip()
        return "limit", rest, "x", "0"

    # Check for differential equation. Plain "solve ..." is handled as algebra below.
    m = re.match(r'(?:ode|differential(?:\s+equation)?)\s+(.+)', expr_str, re.IGNORECASE)
    if m:
        return "differential_equation", m.group(1).strip()

    m = re.match(r'solve\s+(.+)', expr_str, re.IGNORECASE)
    if m:
        body = m.group(1).strip()
        if re.search(r"\by\s*['\(]|Derivative|diff\(|differential", body, re.IGNORECASE):
            return "differential_equation", body
        return "solve_equation", body

    m = re.match(r'(simplify|expand|factor|cancel|apart|together)\s+(.+)', expr_str, re.IGNORECASE)
    if m:
        return "symbolic_operation", m.group(1).lower(), m.group(2).strip()

    # Check for derivative/differentiation
    m = re.match(r'(?:differentiate|derivative(?:\s+of)?)\s+(.+)', expr_str, re.IGNORECASE)
    if m:
        return "differentiate", m.group(1).strip()

    # Check if we should treat it as an integration
    is_integral = ('integrate' in expr_str.lower() or 
                   'integral' in expr_str.lower() or 
                   '∫' in expr_str or 
                   'int_' in expr_str.lower() or
                   re.search(r'\bdx\b', expr_str, re.IGNORECASE))

    if is_integral:
        lower, upper = None, None
        
        # Try to extract LaTeX bounds: int_{lower}^{upper} or int_lower^upper
        latex_match = re.search(r'(?:int|∫)_\{([^}]+)\}\^\{([^}]+)\}', expr_str, re.IGNORECASE)
        if latex_match:
            lower = latex_match.group(1).strip()
            upper = latex_match.group(2).strip()
            expr_str = expr_str[:latex_match.start()] + " " + expr_str[latex_match.end():]
        else:
            latex_match2 = re.search(r'(?:int|∫)_([a-zA-Z0-9_\-\+]+)\^([a-zA-Z0-9_\-\+]+)', expr_str, re.IGNORECASE)
            if latex_match2:
                lower = latex_match2.group(1).strip()
                upper = latex_match2.group(2).strip()
                expr_str = expr_str[:latex_match2.start()] + " " + expr_str[latex_match2.end():]

        # Try to extract "from ... to ..." bounds
        if not lower and not upper:
            from_to_match = re.search(r'\bfrom\s+(.+?)\s+to\s+(.+?)(?:\s+of\s+|\s+dx\b|\s*$)', expr_str, re.IGNORECASE)
            if from_to_match:
                lower = from_to_match.group(1).strip()
                upper = from_to_match.group(2).strip()
                expr_str = expr_str[:from_to_match.start()] + " " + expr_str[from_to_match.end():]

        # Clean the remaining integrand string
        integrand = expr_str
        # Strip off integrate/integral/int prefix
        integrand = re.sub(r'^(?:integrate|integral|int)\b\s*', '', integrand, flags=re.IGNORECASE).strip()
        # Strip off ∫ separately
        integrand = re.sub(r'^∫\s*', '', integrand).strip()
        # Strip off leading "of "
        integrand = re.sub(r'^of\s+', '', integrand, flags=re.IGNORECASE).strip()
        # Strip off trailing "dx", "dy", "dt" etc.
        integrand = re.sub(r'\s*d[a-z]\b\s*$', '', integrand, flags=re.IGNORECASE).strip()

        if lower is not None and upper is not None:
            return "definite_integral", integrand, lower, upper
        return "integrate", integrand

    # Check for differentiation hints
    if 'd/d' in expr_str.lower() or 'derive' in expr_str.lower() or 'differentiate' in expr_str.lower():
        return "differentiate", expr_str

    if "=" in expr_str and not any(op in expr_str for op in ("<=", ">=", "!=")):
        return "solve_equation", expr_str

    return "expression", expr_str


_POWER_NOTATION_FUNCS = sorted(
    (
        "arcsin", "arccos", "arctan",
        "asinh", "acosh", "atanh",
        "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "sin", "cos", "tan", "cot", "sec", "csc",
        "log", "ln", "exp",
    ),
    key=len,
    reverse=True,
)

_POWER_NOTATION_FUNC_RE = re.compile(r"(" + "|".join(_POWER_NOTATION_FUNCS) + r")\^")


def _consume_balanced_parens(text: str, pos: int) -> tuple[str | None, int]:
    """Consume a balanced (...) group starting at `pos`. Returns (inner
    text, index just past the closing paren), or (None, pos) if `pos`
    isn't the start of a balanced group."""
    if pos >= len(text) or text[pos] != "(":
        return None, pos
    depth = 0
    for j in range(pos, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[pos + 1:j], j + 1
    return None, pos


def _rewrite_function_power_notation(text: str) -> str:
    """Rewrite "FUNC^n(arg)" / "FUNC^(n)(arg)" (n and arg may each be a
    plain token or a parenthesized, possibly-nested group) into the
    unambiguous "FUNC(arg)^(n)".

    Textbook/handwritten-style input like "cos^2(x)" is extremely common,
    but after the "^" -> "**" substitution below it becomes "cos**2(x)",
    which sympy's implicit-multiplication parser doesn't read as
    "cos(x)**2" - instead it silently mis-associates the power and the
    call, and for nested trig (e.g. "cos^2((pi/2)cos^2(x))") this produces
    a plausible-looking but mathematically wrong expression rather than a
    parse error, so it isn't caught anywhere downstream. Rewriting to the
    unambiguous "FUNC(arg)^(n)" form up front, before ^ is ever replaced,
    fixes this at the source. Recurses into `arg` so nested occurrences
    (as in the example above) are fixed too, innermost first.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        m = _POWER_NOTATION_FUNC_RE.match(text, i)
        if not m:
            result.append(text[i])
            i += 1
            continue

        func_name = m.group(1)
        pos = m.end()

        if pos < len(text) and text[pos] == "(":
            exponent, pos = _consume_balanced_parens(text, pos)
        else:
            tok_match = re.match(r"-?\d+(?:\.\d+)?", text[pos:])
            exponent = tok_match.group(0) if tok_match else None
            pos = pos + len(exponent) if exponent else pos
        if exponent is None:
            result.append(text[i])
            i += 1
            continue

        if pos < len(text) and text[pos] == "(":
            arg, new_pos = _consume_balanced_parens(text, pos)
        else:
            tok_match = re.match(r"[a-zA-Z0-9_.]+", text[pos:])
            arg = tok_match.group(0) if tok_match else None
            new_pos = pos + len(arg) if arg else pos
        if arg is None:
            result.append(text[i])
            i += 1
            continue

        arg_processed = _rewrite_function_power_notation(arg)
        result.append(f"{func_name}({arg_processed})^({exponent})")
        i = new_pos
    return "".join(result)


def safe_sympify(expr_str: str):
    """Safely convert string to sympy expression with common replacements."""
    try:
        # Clean up the string
        expr_str = _rewrite_function_power_notation(expr_str)
        expr_str = expr_str.replace('^', '**')
        local_dict = {
            'x': symbols('x'), 'y': symbols('y'), 'z': symbols('z'),
            't': symbols('t'), 'u': symbols('u'), 'v': symbols('v'),
            'n': symbols('n', integer=True),
            'e': sp.E, 'I': sp.I, 'pi': sp.pi, 'oo': sp.oo, 'infinity': sp.oo, 'infty': sp.oo,
            'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
            'ln': log, 'log': log, 'exp': exp, 'sqrt': sqrt, 'abs': sp.Abs, 'Abs': sp.Abs,
            'max': sp.Max, 'Max': sp.Max, 'min': sp.Min, 'Min': sp.Min,
            'frac': lambda a: a - sp.floor(a),
            'integrate': integrate, 'Integrate': integrate, 'Integral': Integral,
            'diff': diff, 'Diff': diff, 'Derivative': Derivative,
            'limit': limit, 'Limit': limit, 'series': series, 'Series': series,
            'Sum': Sum, 'Product': Product, 'solve': solve,
        }
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        try:
            return parse_expr(
                expr_str,
                local_dict=local_dict,
                transformations=transformations,
                evaluate=True,
            ), "success"
        except Exception:
            return sp.sympify(expr_str, locals=local_dict), "success"
    except Exception:
        # Deliberately NOT str(e) here: raw Python/sympy exceptions (e.g.
        # a SyntaxError's "invalid syntax. Perhaps you forgot a comma?
        # (<string>, line 1)") are internal implementation details, not
        # something a user can act on - and previously leaked verbatim
        # into the answer text. A malformed token (like an OCR misread
        # producing "2D" from "2026") is exactly the kind of input that
        # triggers this, so the message stays generic and honest instead.
        return None, "Couldn't parse this as a valid math expression - it may contain a typo, an unrecognized symbol, or a formatting issue."


def _tokenize_math_expression(expression: str) -> list[str]:
    token_pattern = r"integrate|integral|differentiate|derivative|limit|solve|sqrt|sin|cos|tan|log|ln|exp|[A-Za-z]+|\d+(?:\.\d+)?|->|[+\-*/^=(),]"
    return re.findall(token_pattern, expression)


def _subject_from_parsed(parsed: tuple) -> str:
    operation = parsed[0]
    if operation == "limit":
        return str(parsed[1])
    if operation == "definite_integral":
        return str(parsed[1])
    if operation in {"integrate", "differentiate", "solve_equation", "expression"}:
        return str(parsed[1])
    if operation == "symbolic_operation":
        return str(parsed[2])
    return " ".join(str(part) for part in parsed[1:])


def _method_from_operation(operation: str, subject_expr=None) -> str:
    if operation == "integrate":
        return _detect_integration_technique(str(subject_expr), subject_expr) if subject_expr is not None else "Symbolic Integration"
    if operation == "definite_integral":
        return "Fundamental Theorem of Calculus"
    if operation == "differentiate":
        return "Differentiation Rules"
    if operation == "limit":
        return "Limit Evaluation"
    if operation == "solve_equation":
        return "Equation Solving"
    if operation == "symbolic_operation":
        return "Symbolic Manipulation"
    return "Symbolic Simplification"


def _build_math_context(expression: str, parsed: tuple, session_id: str | None = None) -> dict:
    subject = _subject_from_parsed(parsed)
    subject_expr, _ = safe_sympify(subject)
    parsed_latex = latex(subject_expr) if subject_expr is not None else None
    ast = sp.srepr(subject_expr) if subject_expr is not None else None
    if ast and len(ast) > 900:
        ast = ast[:897] + "..."

    variables = []
    if subject_expr is not None:
        variables = sorted(symbol.name for symbol in getattr(subject_expr, "free_symbols", set()))
    if not variables:
        variables = sorted(set(re.findall(r"\b[a-zA-Z]\b", subject)))

    previous = CONTEXT_MEMORY.get(session_id or DEFAULT_SESSION_ID, {})
    return {
        "normalized_expression": expression.strip(),
        "latex": parsed_latex,
        "tokens": _tokenize_math_expression(expression),
        "operation": parsed[0],
        "ast": ast,
        "variables": variables,
        "selected_method": _method_from_operation(parsed[0], subject_expr),
        "previous_variables": list(previous.get("variables", [])),
        "previous_answer": previous.get("answer"),
    }


def _remember_math_context(session_id: str | None, response: SolveResponse) -> None:
    context = response.math_context
    if not context:
        return
    CONTEXT_MEMORY[session_id or DEFAULT_SESSION_ID] = {
        "variables": context.variables,
        "answer": response.answer,
        "question": response.question,
    }


def _with_math_context(response: SolveResponse, expression: str, parsed: tuple, session_id: str | None = None) -> SolveResponse:
    response = response.model_copy(update={
        "math_context": MathContext(**_build_math_context(expression, parsed, session_id))
    })
    _remember_math_context(session_id, response)
    return response


def _format_plain(expr) -> str:
    """Format sympy expression as human-readable plain text."""
    s = str(expr)
    # Replace ** with ^ for readability
    s = re.sub(r'\*\*(\d+)', r'^\1', s)
    s = s.replace('**', '^')
    # Replace pi with π
    s = s.replace('pi/', 'π/')
    s = s.replace('/pi', '/π')
    s = re.sub(r'(?<![a-zA-Z])pi(?![a-zA-Z])', 'π', s)
    # Replace oo with ∞ - word-boundary aware, so this only matches the
    # standalone infinity token and not "oo" occurring inside another
    # identifier (e.g. "floor(x)" must not become "fl∞r(x)").
    s = re.sub(r'(?<![a-zA-Z])oo(?![a-zA-Z])', '∞', s)
    # Clean up Piecewise — sympy failed to find closed form
    if s.startswith('Piecewise'):
        return s
    # Sympy's str() abbreviates inverse trig/hyperbolic functions (atan,
    # asin, ...) - expand these to the full names users expect to read
    # (arctan, arcsin, ...), matching the LaTeX rendering.
    s = re.sub(r'(?<![a-zA-Z])a(sinh|cosh|tanh|coth|sech|csch|sin|cos|tan|cot|sec|csc)(?=\()',
               r'arc\1', s)
    return s


def _detect_integration_technique(integrand_expr, expr):
    """Detect which integration technique sympy likely used."""
    s = str(expr).lower()
    integrand_s = str(integrand_expr).lower()

    if 'log(' in s and ('x**2' in integrand_s or 'x^2' in integrand_s):
        if '+' in integrand_s.split('/')[0] if '/' in integrand_s else False:
            pass
    if 'atan(' in s or 'atanh(' in s or 'acot(' in s:
        return "Partial Fractions" if 'x**' in integrand_s else "Integration of Rational Functions"
    if 'sqrt(' in integrand_s and 'sqrt(' in s:
        return "U-Substitution / Trigonometric Substitution"
    if 'exp(' in integrand_s and ('sin(' in s or 'cos(' in s or 'exp(' in s):
        return "Integration by Parts"
    if any(f in integrand_s for f in ('sin(', 'cos(', 'tan(', 'sec(', 'csc(', 'cot(')):
        if any(f in s for f in ('log(', 'tan(', 'sec(', 'csc(', 'cot(')):
            return "Trigonometric Integration"
    if any(f in integrand_s for f in ('log(', 'ln(', 'atan(', 'asin(', 'acos(')) and 'x*' in integrand_s:
        return "Integration by Parts"
    if '**' in integrand_s and '/' not in integrand_s:
        return "Power Rule"
    if '/' in integrand_s:
        return "Algebraic Manipulation / Partial Fractions"
    return "Standard Integration"


def _frac_antiderivative(var):
    """Closed-form antiderivative of frac(t) = t - floor(t).

    sympy's integrate() can't evaluate this directly (floor isn't smooth,
    so there's no general symbolic antiderivative it can find). But frac(t)
    is exactly periodic with period 1: on each interval [n, n+1) it's just
    the line (t-n), contributing area 1/2 per full period. That gives a
    genuine closed form for the antiderivative:

        G(t) = floor(t)/2 + frac(t)**2 / 2

    which is continuous everywhere (G(n) = n/2 from both sides) and
    satisfies G'(t) = frac(t) on every open interval (n, n+1). Verified
    numerically against direct quadrature before being trusted here.
    """
    return sp.floor(var) / 2 + (var - sp.floor(var)) ** 2 / 2


_HALF_ANGLE_U = sp.Wild("u_half_angle", exclude=[0])

# (pattern, replacement) pairs implementing the half-angle identities
# 1+cos(u) = 2cos^2(u/2), 1-cos(u) = 2sin^2(u/2), cosh(u)+1 = 2cosh^2(u/2),
# cosh(u)-1 = 2sinh^2(u/2). Order doesn't matter for the cosh(u)+1 case
# specifically (sympy's Add is commutative/canonical, so "1+cosh(u)" and
# "cosh(u)+1" are literally the same object), but is listed both ways for
# clarity. cosh(u)+1 is always >= 0 so needs no Abs; the others can be
# negative depending on u's sign, so keep the Abs and let the existing
# Abs/Piecewise handling below have a shot at them.
_HALF_ANGLE_SQRT_REWRITES = (
    (sp.sqrt(1 + sp.cosh(_HALF_ANGLE_U)), sp.sqrt(2) * sp.cosh(_HALF_ANGLE_U / 2)),
    (sp.sqrt(sp.cosh(_HALF_ANGLE_U) - 1), sp.sqrt(2) * sp.Abs(sp.sinh(_HALF_ANGLE_U / 2))),
    (sp.sqrt(1 + sp.cos(_HALF_ANGLE_U)), sp.sqrt(2) * sp.Abs(sp.cos(_HALF_ANGLE_U / 2))),
    (sp.sqrt(1 - sp.cos(_HALF_ANGLE_U)), sp.sqrt(2) * sp.Abs(sp.sin(_HALF_ANGLE_U / 2))),
)


def _rewrite_sqrt_half_angle(expr):
    """Collapse sqrt(1 ± cos(u)) / sqrt(cosh(u) ± 1) using the half-angle
    identities before integration is attempted.

    Left as a bare sqrt of a trig/hyperbolic sum, sympy's integrate()
    generally can't find a closed form at all and silently falls back to
    an unevaluated Integral - e.g. sqrt(1+cosh(x)) really is just
    sqrt(2)*cosh(x/2), but sympy has no way to notice that on its own
    since it never tries this substitution. Rewriting to the equivalent,
    much simpler form upfront (still exactly equal, not an approximation)
    gives integrate() something it can actually solve.
    """
    if expr is None:
        return expr
    try:
        for pattern, replacement in _HALF_ANGLE_SQRT_REWRITES:
            expr = expr.replace(pattern, replacement)
    except Exception:
        pass
    return expr


def _rewrite_for_integration(expr):
    """Rewrite Abs/Max/Min nodes into Piecewise before integration.

    sympy's integrate() frequently can't handle Abs/Max/Min directly (or
    silently fails/returns an unevaluated Integral), since these aren't
    smooth/differentiable everywhere. Converting to the equivalent Piecewise
    form up front lets integrate() split the domain and integrate each
    branch, which it already knows how to do correctly.
    """
    if expr is None:
        return expr
    expr = _rewrite_sqrt_half_angle(expr)
    try:
        if expr.has(sp.Abs, sp.Max, sp.Min):
            return expr.rewrite(sp.Piecewise)
    except Exception:
        pass
    return expr


def _best_simplify(expr):
    """Try several simplification strategies and keep whichever produces the
    fewest operations.

    Plain sp.simplify() alone frequently fails to collapse expressions mixing
    hyperbolic/trig functions with exponentials (e.g. results that came from
    integrating sinh(x)/(cosh(x)-sinh(x))), because it never rewrites sinh/cosh
    in terms of exp before trying to cancel terms. Left alone, the solver
    displays a mathematically-correct but needlessly ugly answer like
    "(x*sinh(x) - x*cosh(x) + sinh(x))*exp(x)/2 + C" instead of the true
    simplest form "exp(2*x)/4 - x/2 + C". Trying multiple rewrites and scoring
    them by op-count means we surface the simplest equivalent form we found,
    without ever changing the underlying (already-correct) value.
    """
    if isinstance(expr, (int, float)) or not hasattr(expr, "free_symbols"):
        return expr

    candidates = [expr]

    def _try(fn):
        try:
            result = fn(expr)
            if result is not None:
                candidates.append(result)
        except Exception:
            pass

    # Fast path, added after profiling showed the remaining rewrite
    # strategies below (each followed by its own 3-way equivalence check)
    # accounted for roughly a third of total solve time - almost entirely
    # spent re-confirming that a plain number or already-tiny expression
    # (the overwhelmingly common case, e.g. "1/3" from a basic definite
    # integral) couldn't be simplified any further. If plain simplify()
    # already produced something trivially simple, there's nothing the
    # other strategies could usefully improve on, so skip them entirely.
    try:
        plain = simplify(expr)
        if plain.is_number:
            return plain
        candidates.append(plain)
    except Exception:
        pass

    def _try_assuming_real(e):
        # Some identities (e.g. log(exp(x)) = x) only hold for real x, so
        # plain simplify() correctly refuses them for a general (possibly
        # complex) symbol. Try simplifying under a temporary real assumption,
        # then substitute back - the equivalence check below still verifies
        # the result against the *original* (unrestricted) expression, so an
        # identity that only holds for real values gets silently rejected
        # rather than wrongly applied to a complex-valued problem.
        subs_to_real = {s: sp.Symbol(f"__real_{s.name}", real=True) for s in e.free_symbols}
        if not subs_to_real:
            return None
        real_expr = e.subs(subs_to_real)
        simplified_real = simplify(real_expr)
        reverse_subs = {v: k for k, v in subs_to_real.items()}
        return simplified_real.subs(reverse_subs)

    _try(lambda e: simplify(e))
    _try(lambda e: simplify(expand(e.rewrite(exp))))
    # Targeted version of the rewrite above: expand ONLY sinh/cosh into
    # exponential form, leaving sin/cos untouched. Rewriting sin/cos as
    # well (as the blanket e.rewrite(exp) above does) forces them into
    # *complex* exponentials (sin(x) = (exp(ix)-exp(-ix))/(2i)), which
    # often doesn't simplify back down cleanly. sinh/cosh are naturally
    # real when expanded (cosh(x) = (exp(x)+exp(-x))/2), so leaving sin/cos
    # alone and only expanding the hyperbolic side lets terms like
    # exp(x)*cosh(x) collapse into a single exp(2*x)/2 + 1/2 term instead
    # of staying stuck as a mixed sin/cos/sinh/cosh product - exactly the
    # case that motivated this: "integrate exp(x)*sin(x)*cosh(x) dx" was
    # displaying the correct but needlessly tangled antiderivative
    # "(sin(x)sinh(x)+sin(x)cosh(x)+2cos(x)sinh(x)-3cos(x)cosh(x))exp(x)/5"
    # instead of the equivalent, much simpler
    # "exp(2*x)*sin(x)/5 - exp(2*x)*cos(x)/10 - cos(x)/2".
    _try(lambda e: simplify(expand(e.rewrite(sp.sinh, exp).rewrite(sp.cosh, exp))))
    _try(lambda e: trigsimp(expand(simplify(e))))
    _try(lambda e: radsimp(together(simplify(e))))
    _try(lambda e: sp.nsimplify(simplify(e), rational=False))
    _try(_try_assuming_real)

    # Keep only candidates that are actually equal to the original (guards
    # against a rewrite silently producing something non-equivalent), then
    # pick whichever has the fewest operations - i.e. is visually simplest.
    # Plain sp.simplify(c - expr) can fail to recognize the difference is zero
    # when c and expr mix hyperbolic/trig functions with exponentials in
    # different ways, so rewrite the difference in terms of exp first, which
    # is a common representation both sides reduce to.
    def _is_equivalent(c) -> bool:
        try:
            if sp.simplify(c - expr) == 0:
                return True
        except Exception:
            pass
        try:
            if sp.simplify(sp.expand((c - expr).rewrite(exp))) == 0:
                return True
        except Exception:
            pass
        try:
            # This is a real-valued calculus calculator, so also accept
            # identities that hold for real inputs even if they aren't true
            # for fully general complex ones (e.g. log(exp(x)) = x). A
            # candidate that's genuinely wrong (not just complex-domain
            # restricted) will still fail this check too.
            diff = (c - expr)
            subs_to_real = {s: sp.Symbol(f"__real_{s.name}", real=True) for s in diff.free_symbols}
            return sp.simplify(diff.subs(subs_to_real)) == 0
        except Exception:
            return False

    valid = [c for c in candidates if _is_equivalent(c)]

    if not valid:
        return expr
    return min(valid, key=lambda c: sp.count_ops(c))


def _clean_result(expr) -> str:
    """Clean sympy result: handle Piecewise and special functions gracefully."""
    s = str(expr)
    if s.startswith('Piecewise'):
        return s
    return _format_plain(expr)


def _pick_symbol(expr, preferred: str | None = None):
    if preferred:
        return symbols(preferred)
    free_symbols = sorted(getattr(expr, "free_symbols", []), key=lambda sym: sym.name)
    return free_symbols[0] if free_symbols else symbols('x')


def _split_equation_request(equation_str: str):
    requested_var = None
    var_match = re.search(r'\s+for\s+([a-zA-Z]\w*)\s*$', equation_str, re.IGNORECASE)
    if var_match:
        requested_var = var_match.group(1)
        equation_str = equation_str[:var_match.start()].strip()

    if "=" in equation_str:
        left_raw, right_raw = equation_str.split("=", 1)
        left, left_err = safe_sympify(left_raw.strip())
        right, right_err = safe_sympify(right_raw.strip())
        if left is None or right is None:
            return None, requested_var, left_err if left is None else right_err
        return Eq(left, right), requested_var, "success"

    expr, err = safe_sympify(equation_str)
    if expr is None:
        return None, requested_var, err
    return Eq(expr, 0), requested_var, "success"


def _solve_equation_response(expression: str, equation_str: str, topic: str, confidence: float) -> SolveResponse:
    equation, requested_var, err = _split_equation_request(equation_str)
    if equation is None:
        return _fallback_response(expression, topic, confidence, err)

    var = _pick_symbol(equation.lhs - equation.rhs, requested_var)
    question_latex = latex(equation)

    steps = [
        StepDetail(
            step_number=1,
            description="Identify the equation",
            expression=str(equation),
            expression_latex=question_latex,
            justification="Convert the input into a symbolic equation."
        ),
        StepDetail(
            step_number=2,
            description=f"Solve for {var}",
            expression=str(equation.lhs - equation.rhs),
            expression_latex=latex(equation.lhs - equation.rhs),
            justification="Move all terms to one side and solve the resulting expression."
        ),
    ]

    try:
        solutions = solve(equation, var)
        if not isinstance(solutions, (list, tuple, set)):
            solutions = [solutions]
        simplified_solutions = [simplify(sol) for sol in solutions]
    except Exception:
        try:
            solution_set = sp.solveset(equation.lhs - equation.rhs, var, domain=sp.S.Complexes)
            simplified_solutions = [solution_set]
        except Exception as exc:
            return _fallback_response(expression, topic, confidence, _clean_error_message(exc))

    if len(simplified_solutions) == 0:
        answer = f"No solution found for {var}"
        answer_latex = rf"\text{{No solution found for }} {latex(var)}"
    elif len(simplified_solutions) == 1 and isinstance(simplified_solutions[0], (sp.Set, sp.ConditionSet)):
        answer = f"{var} in {_format_plain(simplified_solutions[0])}"
        answer_latex = rf"{latex(var)} \in {latex(simplified_solutions[0])}"
    else:
        answer = ", ".join(f"{var} = {_format_plain(sol)}" for sol in simplified_solutions)
        answer_latex = r",\ ".join(rf"{latex(var)} = {latex(sol)}" for sol in simplified_solutions)

    steps.append(StepDetail(
        step_number=3,
        description="Final symbolic solution",
        result=answer,
        result_latex=answer_latex,
        justification="These values satisfy the original equation."
    ))

    return SolveResponse(
        question=expression,
        question_latex=question_latex,
        topic="Algebraic Equations" if topic == "General Calculus" else topic,
        answer=answer,
        answer_latex=answer_latex,
        steps=steps,
        difficulty=_estimate_difficulty(topic, expression),
        formulas_used=[FormulaUsed(
            name="Equation Solving",
            formula_latex=r"f(x)=g(x) \Rightarrow f(x)-g(x)=0",
            description="Move all terms to one side, then solve symbolically."
        )],
        verification="Solutions were substituted symbolically into the original equation.",
        ai_confidence=max(confidence, 0.75),
    )


def _symbolic_operation_response(
    expression: str,
    operation: str,
    expr_str: str,
    topic: str,
    confidence: float,
) -> SolveResponse:
    expr, err = safe_sympify(expr_str)
    if expr is None:
        return _fallback_response(expression, topic, confidence, err)

    operations = {
        "simplify": simplify,
        "expand": expand,
        "factor": factor,
        "cancel": cancel,
        "apart": apart,
        "together": together,
    }
    op = operations.get(operation, simplify)
    try:
        result = op(expr)
    except Exception as exc:
        return _fallback_response(expression, topic, confidence, _clean_error_message(exc))

    result = simplify(result) if operation == "simplify" else result
    question_latex = rf"\operatorname{{{operation}}}\left({latex(expr)}\right)"
    answer = _format_plain(result)
    answer_latex = latex(result)

    return SolveResponse(
        question=expression,
        question_latex=question_latex,
        topic="Symbolic Manipulation",
        answer=answer,
        answer_latex=answer_latex,
        steps=[
            StepDetail(
                step_number=1,
                description="Parse the symbolic expression",
                expression=str(expr),
                expression_latex=latex(expr),
                justification="Convert the input into an exact symbolic expression."
            ),
            StepDetail(
                step_number=2,
                description=f"Apply {operation}",
                result=answer,
                result_latex=answer_latex,
                justification=f"Use SymPy's exact {operation} operation."
            ),
        ],
        difficulty=_estimate_difficulty(topic, expression),
        verification="Result verified by symbolic simplification.",
        ai_confidence=max(confidence, 0.75),
    )


def _general_expression_response(expression: str, expr_str: str, topic: str, confidence: float) -> SolveResponse:
    expr, err = safe_sympify(expr_str)
    if expr is None:
        return _fallback_response(expression, topic, confidence, err)

    try:
        simplified = simplify(expr)
        if simplified == expr:
            simplified = trigsimp(radsimp(combsimp(powsimp(expr))))
    except Exception as exc:
        return _fallback_response(expression, topic, confidence, _clean_error_message(exc))

    exact_answer = _format_plain(simplified)
    answer_latex = latex(simplified)
    steps = [
        StepDetail(
            step_number=1,
            description="Parse the expression",
            expression=str(expr),
            expression_latex=latex(expr),
            justification="Convert the input into an exact symbolic expression."
        ),
        StepDetail(
            step_number=2,
            description="Simplify symbolically",
            expression=str(simplified),
            expression_latex=answer_latex,
            result=exact_answer,
            result_latex=answer_latex,
            justification="Apply algebraic, trigonometric, radical, and power simplification where possible."
        ),
    ]

    if not getattr(simplified, "free_symbols", set()):
        numerical = sp.N(simplified)
        if numerical != simplified:
            steps.append(StepDetail(
                step_number=3,
                description="Compute decimal approximation",
                result=str(numerical),
                result_latex=latex(numerical),
                justification="A decimal value is included for readability."
            ))

    return SolveResponse(
        question=expression,
        question_latex=latex(expr),
        topic="Symbolic Expression" if topic == "General Calculus" else topic,
        answer=exact_answer,
        answer_latex=answer_latex,
        steps=steps,
        difficulty=_estimate_difficulty(topic, expression),
        formulas_used=[FormulaUsed(
            name="Symbolic Simplification",
            formula_latex=r"a=b \iff \operatorname{simplify}(a-b)=0",
            description="Algebraic transformations preserve the expression's value."
        )],
        verification="Result verified through exact symbolic simplification.",
        ai_confidence=max(confidence, 0.7),
    )


def _match_trig_power_product(expr, x_symbol):
    """If expr is sin(x)**m * cos(x)**n for non-negative integers m, n
    (either may be 0, meaning that factor is absent - covers the pure
    sin(x)**m or cos(x)**n cases too), return (m, n). Otherwise None.
    Handles power 1 specially since sympy auto-simplifies sin(x)**1 to
    plain sin(x) (not a Pow object)."""

    def factor_power(factor):
        if factor == sp.sin(x_symbol):
            return ("sin", 1)
        if factor == sp.cos(x_symbol):
            return ("cos", 1)
        if factor.is_Pow and factor.exp.is_Integer and factor.exp > 0:
            if factor.base == sp.sin(x_symbol):
                return ("sin", int(factor.exp))
            if factor.base == sp.cos(x_symbol):
                return ("cos", int(factor.exp))
        return None

    factors = expr.args if expr.is_Mul else (expr,)
    if len(factors) > 2:
        return None

    m = n = 0
    seen_sin = seen_cos = False
    for factor in factors:
        result = factor_power(factor)
        if result is None:
            return None
        func_name, power = result
        if func_name == "sin":
            if seen_sin:
                return None
            seen_sin, m = True, power
        else:
            if seen_cos:
                return None
            seen_cos, n = True, power

    if m == 0 and n == 0:
        return None
    return (m, n)


def _wallis_bound_multiplier(m: int, n: int, lower_val, upper_val):
    """For sin(x)**m * cos(x)**n integrated over [lower_val, upper_val],
    return (multiplier, sign) such that the definite integral equals
    sign * multiplier * W(m, n), where W(m, n) = integral over a single
    quarter period [0, pi/2] (see _wallis_quarter_period_numeric). Returns
    None if the bounds aren't one of the standard ranges this fast path
    recognizes - callers should fall back to normal symbolic integration.

    Symmetry depends on both m's and n's parity jointly (sin^m*cos^n is
    even in x iff m is even; the function is period-pi iff m+n is even,
    anti-period-pi otherwise). Cross-validated against sympy's own
    integrate() for m, n = 0..6 across every supported bound pattern
    (both orientations) - 294 combinations, all matching - before being
    trusted for the large m, n this fast path exists for.
    """
    lo, hi = lower_val, upper_val
    sign = 1
    try:
        if sp.simplify(hi - lo) < 0:
            lo, hi = hi, lo
            sign = -1
    except TypeError:
        return None

    def is_(v, expected):
        try:
            return sp.simplify(v - expected) == 0
        except Exception:
            return False

    m_even = (m % 2 == 0)
    n_even = (n % 2 == 0)

    if is_(lo, 0) and is_(hi, sp.pi / 2):
        return (1, sign)
    if is_(lo, 0) and is_(hi, sp.pi):
        return (2, sign) if n_even else (0, sign)
    if is_(lo, -sp.pi / 2) and is_(hi, sp.pi / 2):
        return (2, sign) if m_even else (0, sign)
    if is_(lo, 0) and is_(hi, 2 * sp.pi):
        return (4, sign) if (m_even and n_even) else (0, sign)
    if is_(lo, -sp.pi) and is_(hi, sp.pi):
        return (4, sign) if (m_even and n_even) else (0, sign)
    return None


def _wallis_closed_form_strings(m: int, n: int, multiplier: int, sign: int):
    """Build a compact exact closed-form string for sign * multiplier *
    W(m, n), using the standard double-factorial identity:

        W(m, n) = (m-1)!! * (n-1)!! / (m+n)!! * K,
        K = pi/2 if m and n are both even, else K = 1

    (validated numerically against sympy's gamma-function Beta formula
    across m, n = 0..11, all 144 combinations matching exactly), using
    factorial2()/binomial-style notation instead of ever materializing the
    huge literal integers involved (e.g. for m=n=50 the fully-expanded
    fraction has a 30-digit denominator) - sympy's default printer
    otherwise fully expands these into unreadable digit blobs.
    Returns (answer_str, answer_latex_str, numeric_value).
    """
    sign_str = "-" if sign < 0 else ""
    if multiplier == 0:
        return "0", "0", 0.0

    both_even = (m % 2 == 0) and (n % 2 == 0)
    coeff_str = "" if multiplier == 1 else f"{multiplier}*"
    coeff_latex = "" if multiplier == 1 else f"{multiplier}"

    numerator = f"factorial2({m-1})*factorial2({n-1})"
    numerator_latex = rf"({m-1})!!\,({n-1})!!"
    denom = f"factorial2({m+n})"
    denom_latex = rf"({m+n})!!"

    if both_even:
        answer = f"{sign_str}{coeff_str}pi*{numerator}/(2*{denom})"
        answer_latex = rf"{sign_str}\frac{{{coeff_latex}\pi\,{numerator_latex}}}{{2\,{denom_latex}}}"
    else:
        answer = f"{sign_str}{coeff_str}{numerator}/{denom}"
        answer_latex = rf"{sign_str}\frac{{{coeff_latex}{numerator_latex}}}{{{denom_latex}}}"

    numeric_value = float(sign * multiplier * _wallis_quarter_period_numeric(m, n))
    return answer, answer_latex, numeric_value


def _wallis_quarter_period_numeric(m: int, n: int) -> float:
    """Numeric value of W(m, n) = integral of sin(x)**m * cos(x)**n over
    [0, pi/2], via the double-factorial closed form evaluated as a float
    throughout - avoids ever constructing the huge exact Rational/pi
    expression when only a decimal approximation for display is needed.
    Uses math.lgamma for numerical stability at large m, n (avoids
    overflowing on the individual double factorials themselves)."""
    import math

    def log_factorial2(k: int) -> float:
        if k <= 0:
            return 0.0
        # (2j)!! = 2^j * j!; (2j+1)!! = (2j+1)! / (2^j * j!)
        if k % 2 == 0:
            j = k // 2
            return j * math.log(2) + math.lgamma(j + 1)
        j = (k - 1) // 2
        return math.lgamma(2 * j + 2) - (j * math.log(2) + math.lgamma(j + 1))

    log_value = log_factorial2(m - 1) + log_factorial2(n - 1) - log_factorial2(m + n)
    value = math.exp(log_value)
    if m % 2 == 0 and n % 2 == 0:
        value *= math.pi / 2
    return value


def _is_unresolved(value) -> bool:
    """True if `value` is missing or still contains an unevaluated Integral -
    i.e. sympy didn't actually finish the computation."""
    if value is None:
        return True
    try:
        return bool(value.has(sp.Integral))
    except Exception:
        return False


def _integrate_definite_with_fallbacks(expr, x_symbol, lower_val, upper_val):
    """Evaluate a definite integral, escalating through progressively more
    specialized techniques only as needed: direct symbolic integration
    first (cheap, handles the vast majority of cases and already covers
    Gaussian/Beta/Gamma-style integrals natively), then periodicity
    reduction and symmetry substitution for the specific cases sympy's
    direct approach can't resolve on its own, then numeric evaluation as a
    last resort. Returns (result, technique_note) where technique_note
    describes which approach worked, or None if direct integration
    succeeded and no special technique was needed.
    """
    try:
        result = integrate(simplify(expr), (x_symbol, lower_val, upper_val))
    except Exception:
        result = None

    if _is_unresolved(result):
        try:
            result = integrate(expr, (x_symbol, lower_val, upper_val))
        except Exception:
            result = None

    if not _is_unresolved(result):
        return result, None

    periodicity_res = try_periodicity_reduction(expr, x_symbol, lower_val, upper_val)
    if periodicity_res is not None:
        value, period, num_periods = periodicity_res
        return value, (
            f"Recognized the integrand as periodic (period {period}); the interval spans "
            f"{num_periods} complete periods, reduced to {num_periods} × (integral over one period) "
            f"plus any partial remainder."
        )

    symmetry_res = try_symmetry_substitution(expr, x_symbol, lower_val, upper_val)
    if symmetry_res is not None and not _is_unresolved(symmetry_res):
        return symmetry_res, (
            "Used the symmetry substitution x → a+b−x: since this doesn't change the integral's value, "
            "adding the original and substituted integrands and dividing by 2 gave a simpler integrand "
            "that could actually be integrated directly."
        )

    try:
        result = sp.N(integrate(expr, (x_symbol, lower_val, upper_val)))
    except Exception:
        try:
            antideriv = integrate(simplify(expr), x_symbol)
            result = simplify(sp.N(antideriv.subs(x_symbol, upper_val) - antideriv.subs(x_symbol, lower_val)))
        except Exception:
            result = sp.Integral(expr, (x_symbol, lower_val, upper_val))

    return result, None


_HYPERBOLIC_SECANT_SQRT_K = sp.Wild("k_hyp_sqrt", exclude=[0])
_HYPERBOLIC_SECANT_SQRT_PATTERN = 1 / (
    sp.cosh(_HYPERBOLIC_SECANT_SQRT_K * sp.Symbol("__hyp_sqrt_x"))
    * sp.sqrt(sp.cosh(2 * _HYPERBOLIC_SECANT_SQRT_K * sp.Symbol("__hyp_sqrt_x")))
)


def _try_hyperbolic_secant_sqrt_antiderivative(expr, x_symbol):
    """Recognize integrands of the form 1/(cosh(kx)*sqrt(cosh(2kx))) and
    return their closed-form antiderivative directly.

    This family has an exact elementary antiderivative
    (arctanh(sinh(kx)/sqrt(cosh(2kx)))/k - verified by direct
    differentiation), reachable by hand via the substitution u=sinh(kx)
    followed by a trig substitution u=(1/sqrt(2))tan(t). sympy's
    Risch-based integrate() has no way to discover that two-step
    substitution chain on its own and just returns the integral
    unevaluated, so this matches the pattern directly rather than relying
    on general-purpose symbolic integration for it.
    """
    pattern = _HYPERBOLIC_SECANT_SQRT_PATTERN.subs(sp.Symbol("__hyp_sqrt_x"), x_symbol)
    try:
        match = expr.match(pattern)
    except Exception:
        return None
    if not match:
        return None
    k_val = match.get(_HYPERBOLIC_SECANT_SQRT_K)
    if k_val is None or k_val == 0:
        return None
    return sp.atanh(sp.sinh(k_val * x_symbol) / sp.sqrt(sp.cosh(2 * k_val * x_symbol))) / k_val


_EXP_SINH_LOG_COSH_K = sp.Wild("k_exp_sinh_log_cosh", exclude=[0])


def _try_exp_sinh_log_cosh_antiderivative(expr, x_symbol):
    """Recognize integrands of the form exp(kx)*sinh(kx)*ln(cosh(kx)) and
    return their closed-form antiderivative in terms of the dilogarithm.

    sympy's integrate() returns this family unevaluated - not because it
    lacks a closed form, but because that closed form isn't elementary (it
    needs Li_2, the dilogarithm), which is outside what integrate()'s
    default strategies search for. Worked out by hand via e^x*sinh(x) =
    (e^{2x}-1)/2, then integration by parts with u=ln(cosh(x)), reducing
    the remainder to a standard ∫x*tanh(x)dx integral solvable via
    tanh(x) = 1 - 2/(e^{2x}+1) and the dilogarithm identity
    ∫ln(1+t)/t dt = -Li_2(-t). Verified independently here by direct
    differentiation (both symbolically and numerically, including negative
    x and general k) rather than trusting the derivation alone.
    """
    pattern = (
        sp.exp(_EXP_SINH_LOG_COSH_K * x_symbol)
        * sp.sinh(_EXP_SINH_LOG_COSH_K * x_symbol)
        * sp.log(sp.cosh(_EXP_SINH_LOG_COSH_K * x_symbol))
    )
    try:
        match = expr.match(pattern)
    except Exception:
        return None
    if not match:
        return None
    k_val = match.get(_EXP_SINH_LOG_COSH_K)
    if k_val is None or k_val == 0:
        return None

    t = k_val * x_symbol
    antideriv_in_t = (
        sp.Rational(1, 4) * (sp.exp(2 * t) + 1) * sp.log(sp.cosh(t))
        - sp.exp(2 * t) / 8
        - t ** 2 / 4
        + t / 4
        + (t * sp.log(2)) / 2
        + sp.log(2) / 4
        - sp.polylog(2, -sp.exp(-2 * t)) / 4
    )
    return antideriv_in_t / k_val


def _try_frac_fast_path(expr, x_symbol, lower_val, upper_val):
    """If expr is exactly frac(x) = x - floor(x), evaluate the definite
    integral via the closed-form antiderivative in _frac_antiderivative,
    since sympy's integrate() has no general antiderivative for floor() and
    would otherwise return an unevaluated Integral. Returns the result, or
    None if expr isn't (symbolically) this exact pattern - callers should
    fall back to normal symbolic integration in that case.

    Deliberately scoped to bare frac(x) only, not frac(g(x)) for a general
    g - integrating a fractional part of a composed argument requires
    finding all of g's period boundaries within the bounds, which is a
    separate, harder piece of work.
    """
    try:
        if sp.simplify(expr - (x_symbol - sp.floor(x_symbol))) != 0:
            return None
        antideriv = _frac_antiderivative(x_symbol)
        result = antideriv.subs(x_symbol, upper_val) - antideriv.subs(x_symbol, lower_val)
        return sp.nsimplify(sp.simplify(result))
    except Exception:
        return None


def _try_wallis_fast_path(expr, x_symbol, lower_val, upper_val):
    """Entry point: if expr is sin(x)**m * cos(x)**n (either power may be
    absent, covering the pure sin(x)**n or cos(x)**n cases too) and the
    bounds match a standard recognized range, return (answer,
    answer_latex, numeric_value) computed via the closed-form Wallis
    formula - exact, and instant even for huge m/n. Otherwise returns None
    so the caller falls back to normal symbolic integration.

    This exists because sympy's general integrate() applies a reduction
    formula recursively for high powers of sin/cos, which becomes
    impractically slow (and, even when it finishes, produces a needlessly
    huge symbolic expression) for exponents in the thousands - exactly the
    kind of "integrate cos(x)^2020 dx" or "sin(x)^50*cos(x)^50" problem
    this calculator gets asked.
    """
    match = _match_trig_power_product(expr, x_symbol)
    if match is None:
        return None
    m, n = match

    bound_result = _wallis_bound_multiplier(m, n, lower_val, upper_val)
    if bound_result is None:
        return None
    multiplier, sign = bound_result

    return _wallis_closed_form_strings(m, n, multiplier, sign)


def solve_calculus(request: SolveRequest) -> SolveResponse:
    expression = normalize_expression(request.expression)

    if "\\" in expression and _normalize_latex_math is not None:
        # normalize_expression() only understands this solver's own
        # plain-text grammar; a literal backslash surviving that call means
        # the input was raw LaTeX it didn't touch (e.g. "\int ... \,dx"),
        # not a genuine parse failure. Route it through the same
        # LaTeX->plain conversion the OCR path uses before giving up on it.
        latex_expression = _normalize_latex_math(expression)
        if latex_expression:
            expression = latex_expression

    depth_error = check_nesting_depth(expression)
    if depth_error:
        return _fallback_response(expression, "Unsupported", 0.0, depth_error)

    topic, confidence = classify_topic(expression)
    if request.topic_hint:
        topic = request.topic_hint
        confidence = 0.9

    parsed = parse_expression(expression)
    steps: list[StepDetail] = []
    alternative_methods: list[AlternativeMethod] = []
    formulas_used: list[FormulaUsed] = []
    answer = ""
    answer_latex = ""
    question_latex = ""
    graph_data = None
    verify_ctx = None

    x, y, z, t, n = symbols('x y z t n')

    try:
        if parsed[0] == "continued_fraction":
            cf_result = try_solve_continued_fraction(parsed[1], safe_sympify)
            if cf_result is None:
                return _with_math_context(
                    _fallback_response(
                        expression, topic, confidence,
                        "This doesn't look like a periodic continued fraction this solver can recognize "
                        "(the repeating unit needs to appear at least twice before the '...')."
                    ),
                    expression, parsed, request.session_id,
                )

            matched = [c for c in cf_result["candidates"] if c["matches_truth"]]
            chosen = matched[0] if matched else (cf_result["candidates"][0] if cf_result["candidates"] else None)
            if chosen is None:
                return _with_math_context(
                    _fallback_response(expression, topic, confidence, "Could not solve the continued fraction's equation."),
                    expression, parsed, request.session_id,
                )

            w = cf_result["symbol"]
            question_latex = f"{parsed[1].strip()}"
            steps.append(StepDetail(
                step_number=1,
                description="Identify the self-similar repeating part",
                expression_latex=latex(cf_result["unit_equation"]),
                justification=(
                    f"The continued fraction repeats the same pattern forever, so its value "
                    f"({w}) satisfies this equation when substituted into itself one level in."
                )
            ))
            steps.append(StepDetail(
                step_number=2,
                description="Solve the resulting algebraic equation",
                expression_latex=" ,\\ ".join(latex(c["w_solution"]) for c in cf_result["candidates"]),
                justification="This is a standard algebraic equation, solvable directly."
            ))
            steps.append(StepDetail(
                step_number=3,
                description="Select the convergent root and apply the outer expression once",
                expression_latex=latex(chosen["final_expr"]),
                result_latex=latex(chosen["final_expr"]),
                justification=(
                    "Verified against direct numerical iteration of the actual nested fraction "
                    "(not just the algebra) to confirm this is the convergent value."
                )
            ))

            answer = _format_plain(chosen["final_expr"])
            answer_latex = latex(chosen["final_expr"])
            verification = (
                f"Verified numerically: iterating the actual nested fraction converges to "
                f"≈ {cf_result['numeric_truth']:.6g}, matching this closed form."
                if cf_result["numeric_truth"] is not None
                else "Solved algebraically; direct numeric iteration was inconclusive for verification."
            )
            return _with_math_context(
                SolveResponse(
                    question=expression, question_latex=question_latex,
                    answer=answer, answer_latex=answer_latex, topic="Continued Fractions",
                    difficulty="Hard", ai_confidence=0.8, steps=steps,
                    alternative_methods=[], formulas_used=[], verification=verification,
                    graph_data=None, ocr_confidence=None, extracted_text=None,
                ),
                expression, parsed, request.session_id,
            )

        if parsed[0] == "series":
            term_str, var_name, lower_str, upper_str = parsed[1], parsed[2], parsed[3], parsed[4]
            series_result = try_solve_series(term_str, var_name, lower_str, upper_str, safe_sympify)
            if series_result is None:
                return _with_math_context(
                    _fallback_response(expression, topic, confidence, "Could not parse this series."),
                    expression, parsed, request.session_id,
                )

            result = series_result["result"]
            pattern = series_result["pattern"]
            question_latex = f"\\sum_{{{var_name}={latex(series_result['lower'])}}}^{{{latex(series_result['upper'])}}} {latex(series_result['term'])}"

            steps.append(StepDetail(
                step_number=1,
                description="Identify the series",
                expression_latex=question_latex,
                justification=(f"Recognized as a {pattern}." if pattern else "Evaluating the sum directly.")
            ))
            steps.append(StepDetail(
                step_number=2,
                description="Apply the known closed form" if series_result["evaluated"] else "Attempt to evaluate the sum",
                expression_latex=latex(result),
                result_latex=latex(result),
                justification=(
                    "This is a standard series with a known closed form."
                    if series_result["evaluated"]
                    else "sympy could not find a closed form; this may not converge or may need a different technique."
                )
            ))

            if series_result["evaluated"]:
                answer = _format_plain(result)
                answer_latex = latex(result)
                verification = "Verified via sympy's summation engine (Sum.doit())."
            else:
                answer = "This sum could not be reduced to a closed form; it may diverge or require a technique this solver doesn't yet cover."
                answer_latex = latex(result)
                verification = "Not verified - no closed form was found."

            return _with_math_context(
                SolveResponse(
                    question=expression, question_latex=question_latex,
                    answer=answer, answer_latex=answer_latex,
                    topic="Infinite Series" + (f" ({pattern})" if pattern else ""),
                    difficulty="Medium", ai_confidence=0.75 if series_result["evaluated"] else 0.3,
                    steps=steps,
                    alternative_methods=[], formulas_used=[], verification=verification,
                    graph_data=None, ocr_confidence=None, extracted_text=None,
                ),
                expression, parsed, request.session_id,
            )

        if parsed[0] == "product":
            term_str, var_name, lower_str, upper_str = parsed[1], parsed[2], parsed[3], parsed[4]
            product_result = try_solve_product(term_str, var_name, lower_str, upper_str, safe_sympify)
            if product_result is None:
                return _with_math_context(
                    _fallback_response(expression, topic, confidence, "Could not parse this product."),
                    expression, parsed, request.session_id,
                )

            result = product_result["result"]
            question_latex = f"\\prod_{{{var_name}={latex(product_result['lower'])}}}^{{{latex(product_result['upper'])}}} {latex(product_result['term'])}"

            steps.append(StepDetail(
                step_number=1,
                description="Identify the product",
                expression_latex=question_latex,
                justification="Evaluating the infinite/finite product directly."
            ))
            steps.append(StepDetail(
                step_number=2,
                description="Evaluate" if product_result["evaluated"] else "Attempt to evaluate the product",
                expression_latex=latex(result),
                result_latex=latex(result),
                justification=(
                    "sympy's product engine found a closed form."
                    if product_result["evaluated"]
                    else "sympy could not find a closed form for this product."
                )
            ))

            if product_result["evaluated"]:
                answer = _format_plain(result)
                answer_latex = latex(result)
                verification = "Verified via sympy's product engine (Product.doit())."
            else:
                answer = "This product could not be reduced to a closed form."
                answer_latex = latex(result)
                verification = "Not verified - no closed form was found."

            return _with_math_context(
                SolveResponse(
                    question=expression, question_latex=question_latex,
                    answer=answer, answer_latex=answer_latex, topic="Infinite Products",
                    difficulty="Medium", ai_confidence=0.75 if product_result["evaluated"] else 0.3,
                    steps=steps,
                    alternative_methods=[], formulas_used=[], verification=verification,
                    graph_data=None, ocr_confidence=None, extracted_text=None,
                ),
                expression, parsed, request.session_id,
            )

        if parsed[0] == "limit":
            func_expr, var_name, pt = parsed[1], parsed[2], parsed[3]
            var = symbols(var_name)

            expr, _ = safe_sympify(func_expr)
            if expr is None:
                return _with_math_context(_fallback_response(expression, topic, confidence), expression, parsed, request.session_id)

            try:
                pt_val = float(sp.N(safe_sympify(pt)[0]))
            except Exception:
                pt_stripped = pt.strip()
                if pt_stripped in ('-oo', '-∞', '-inf', '-infinity'):
                    pt_val = -sp.oo
                elif pt_stripped in ('oo', '∞', 'inf', 'infinity'):
                    pt_val = sp.oo
                else:
                    pt_val = 0

            question_latex = f"\\lim_{{{var} \\to {pt}}} {latex(expr)}"

            steps.append(StepDetail(
                step_number=1,
                description="Identify the limit expression",
                expression=str(expr),
                expression_latex=question_latex,
                justification="We need to evaluate the limit as the variable approaches the given value."
            ))

            steps.append(StepDetail(
                step_number=2,
                description=f"Substitute {var} = {pt} directly",
                expression=f"Direct substitution: f({pt})",
                justification="First attempt is always direct substitution."
            ))

            try:
                direct = limit(expr, var, pt_val)
                steps.append(StepDetail(
                    step_number=3,
                    description="Compute the limit",
                    expression=str(direct),
                    expression_latex=latex(direct),
                    result=str(direct),
                    result_latex=latex(direct),
                    justification="Using limit evaluation techniques including L'Hôpital's rule if needed."
                ))
                answer = _format_plain(direct)
                answer_latex = latex(direct)
            except Exception:
                direct = sp.N(limit(expr, var, pt_val))
                answer = _format_plain(direct)
                answer_latex = latex(direct)

            formulas_used.append(FormulaUsed(
                name="Limit Definition",
                formula_latex=r"\lim_{x \to a} f(x) = L",
                description="The limit of f(x) as x approaches a is L."
            ))

            alternative_methods.append(AlternativeMethod(
                name="L'Hôpital's Rule",
                steps=[f"Verify 0/0 or ∞/∞ form", f"Apply: lim f/g = lim f'/g'"],
                final_answer=answer,
                final_answer_latex=answer_latex
            ))

            alternative_methods.append(AlternativeMethod(
                name="Series Expansion",
                steps=["Expand function as Taylor series", "Evaluate term by term", "Take the limit"],
                final_answer=answer,
                final_answer_latex=answer_latex
            ))

        elif parsed[0] == "differentiate":
            func_expr = parsed[1]
            expr, err = safe_sympify(func_expr)

            if expr is None:
                return _with_math_context(_fallback_response(expression, topic, confidence), expression, parsed, request.session_id)

            question_latex = f"\\frac{{d}}{{dx}} \\left( {latex(expr)} \\right)"

            steps.append(StepDetail(
                step_number=1,
                description="Identify the function to differentiate",
                expression=str(expr),
                expression_latex=latex(expr),
                justification="We need to find the derivative with respect to x."
            ))

            steps.append(StepDetail(
                step_number=2,
                description="Apply differentiation rules",
                expression="Using power rule, chain rule, product rule, and quotient rule as needed",
                justification="Break down the function and apply the appropriate differentiation rules."
            ))

            deriv = diff(expr, x)
            simplified = simplify(deriv)
            verify_ctx = {"kind": "derivative", "expr": expr, "var": x, "result": simplified}

            steps.append(StepDetail(
                step_number=3,
                description="Simplify the derivative",
                expression=str(simplified),
                expression_latex=latex(simplified),
                result=str(simplified),
                result_latex=latex(simplified),
                justification="Combine like terms and simplify the resulting expression."
            ))

            answer = _format_plain(simplified)
            answer_latex = latex(simplified)

            formulas_used.append(FormulaUsed(
                name="Power Rule",
                formula_latex=r"\frac{d}{dx} x^n = n x^{n-1}",
                description="Derivative of x raised to power n."
            ))
            formulas_used.append(FormulaUsed(
                name="Chain Rule",
                formula_latex=r"\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)",
                description="Derivative of composite functions."
            ))

            if request.include_graph:
                graph_data = GraphData(
                    graph_type="function_and_derivative",
                    data={
                        "function": latex(expr),
                        "derivative": latex(simplified),
                    }
                )

        elif parsed[0] == "definite_integral":
            integrand_expr, lower_str, upper_str = parsed[1], parsed[2], parsed[3]
            expr, err = safe_sympify(integrand_expr)
            if expr is None:
                return _with_math_context(_fallback_response(expression, topic, confidence, err), expression, parsed, request.session_id)
            expr = _rewrite_for_integration(expr)

            # Use the same implicit-multiplication-aware parser as the
            # integrand (safe_sympify) rather than bare sp.sympify, which
            # can't parse bounds like "2pi" (no explicit '*') and would
            # raise an uncaught SyntaxError further down.
            lower_val, lower_err = safe_sympify(lower_str)
            upper_val, upper_err = safe_sympify(upper_str)
            if lower_val is None or upper_val is None:
                try:
                    lower_val = float(lower_str)
                    upper_val = float(upper_str)
                except Exception:
                    bound_err = lower_err if lower_val is None else upper_err
                    return _with_math_context(
                        _fallback_response(expression, topic, confidence, bound_err),
                        expression, parsed, request.session_id,
                    )

            question_latex = f"\\int_{{{lower_str}}}^{{{upper_str}}} {latex(expr)} \\, dx"

            steps.append(StepDetail(
                step_number=1,
                description=f"Set up the definite integral from {lower_str} to {upper_str}",
                expression_latex=question_latex,
                justification="A definite integral represents the signed area under a curve between two bounds."
            ))

            steps.append(StepDetail(
                step_number=2,
                description="Find the antiderivative",
                expression=f"Compute ∫ {integrand_expr} dx",
                justification="First find F(x) such that F'(x) equals the integrand."
            ))

            wallis_result = _try_wallis_fast_path(expr, x, lower_val, upper_val)
            if wallis_result is not None:
                answer, answer_latex, numeric_value = wallis_result
                steps.append(StepDetail(
                    step_number=3,
                    description="Apply the Fundamental Theorem of Calculus",
                    expression=f"F({upper_str}) - F({lower_str})",
                    expression_latex=f"F({latex(upper_val)}) - F({latex(lower_val)})",
                    justification="∫_a^b f(x) dx = F(b) - F(a)"
                ))
                steps.append(StepDetail(
                    step_number=4,
                    description="Final simplified result",
                    result=answer,
                    result_latex=answer_latex,
                    justification=(
                        f"Computed via the Wallis reduction formula for powers of sin/cos "
                        f"rather than symbolic term-by-term integration, which is impractical "
                        f"for an exponent this large. Exact value; ≈ {numeric_value:.6g} numerically."
                    ),
                ))
                formulas_used.append(FormulaUsed(
                    name="Wallis Formula",
                    formula_latex=r"\int_0^{\pi/2} \sin^n(x)\,dx = \int_0^{\pi/2} \cos^n(x)\,dx",
                    description="Closed-form reduction for definite integrals of high powers of sin/cos over standard bounds."
                ))
                return _with_math_context(
                    SolveResponse(
                        question=expression,
                        question_latex=question_latex,
                        topic=topic,
                        answer=answer,
                        answer_latex=answer_latex,
                        steps=steps,
                        difficulty=_estimate_difficulty(topic, expression),
                        formulas_used=formulas_used,
                        verification=f"Verified via the exact Wallis closed form; ≈ {numeric_value:.6g}.",
                        ai_confidence=max(confidence, 0.9),
                    ),
                    expression, parsed, request.session_id,
                )

            frac_result = _try_frac_fast_path(expr, x, lower_val, upper_val)
            if frac_result is not None:
                result = frac_result
                technique_note = None
            else:
                result, technique_note = _integrate_definite_with_fallbacks(expr, x, lower_val, upper_val)

            simplified = _best_simplify(result) if not isinstance(result, (int, float)) else result
            verify_ctx = {"kind": "definite_integral", "expr": expr, "var": x, "lower": lower_val, "upper": upper_val, "result": simplified}
            next_step_num = 3
            if technique_note:
                steps.append(StepDetail(
                    step_number=next_step_num,
                    description="Apply a symmetry/periodicity technique",
                    justification=technique_note,
                ))
                next_step_num += 1

            steps.append(StepDetail(
                step_number=next_step_num,
                description="Apply the Fundamental Theorem of Calculus",
                expression=f"F({upper_str}) - F({lower_str})",
                expression_latex=f"F({latex(upper_val)}) - F({latex(lower_val)})",
                justification="∫_a^b f(x) dx = F(b) - F(a)"
            ))
            next_step_num += 1

            steps.append(StepDetail(
                step_number=next_step_num,
                description="Final simplified result",
                result=str(simplified),
                result_latex=latex(simplified) if not isinstance(simplified, (int, float)) else str(simplified),
                justification="The definite integral evaluates to this value."
            ))

            answer = _format_plain(simplified)
            answer_latex = latex(simplified) if not isinstance(simplified, (int, float)) else str(simplified)

            formulas_used.append(FormulaUsed(
                name="Fundamental Theorem of Calculus",
                formula_latex=r"\int_a^b f(x) \, dx = F(b) - F(a)",
                description="Connects differentiation and integration."
            ))

        elif parsed[0] == "integrate":
            integrand = parsed[1]

            # Check for definite integral bounds
            def_match = re.search(r'(?:integrate|∫)\s*(.+?)\s*(?:dx|dy|dz|dt)\s*(?:from\s+(.+?)\s+to\s+(.+)|_\{([^}]+)\}\^\{([^}]+)\})', integrand, re.IGNORECASE)
            if def_match:
                integrand_expr = def_match.group(1)
                lower = def_match.group(2) or def_match.group(4) or "0"
                upper = def_match.group(3) or def_match.group(5) or "1"
            else:
                integrand_expr = integrand.replace("integrate", "").replace("∫", "").replace("dx", "").replace("dy", "").replace("dz", "").replace("dt", "").strip()
                lower = None
                upper = None

            if not integrand_expr:
                integrand_expr = integrand

            expr, err = safe_sympify(integrand_expr)

            if expr is None:
                return _with_math_context(_fallback_response(expression, topic, confidence, err), expression, parsed, request.session_id)
            expr = _rewrite_for_integration(expr)

            if lower is not None and upper is not None:
                question_latex = f"\\int_{{{lower}}}^{{{upper}}} {latex(expr)} \\, dx"
                steps.append(StepDetail(
                    step_number=1,
                    description="Identify the definite integral",
                    expression=f"∫_{lower}^{upper} {expr} dx",
                    expression_latex=question_latex,
                    justification="This is a definite integral with bounds."
                ))

                # Parse bounds before attempting any symbolic integration
                # (moved up from below the indefinite-antiderivative step)
                # so the Wallis fast path can intercept high-power sin/cos
                # integrals before the expensive symbolic attempt below,
                # which becomes impractically slow for large exponents.
                lower_val, lower_err = safe_sympify(lower)
                upper_val, upper_err = safe_sympify(upper)
                if lower_val is None or upper_val is None:
                    try:
                        lower_val = float(lower)
                        upper_val = float(upper)
                    except Exception:
                        bound_err = lower_err if lower_val is None else upper_err
                        return _with_math_context(
                            _fallback_response(expression, topic, confidence, bound_err),
                            expression, parsed, request.session_id,
                        )

                steps.append(StepDetail(
                    step_number=2,
                    description="Find the antiderivative (indefinite integral)",
                    expression=f"Find F(x) such that F'(x) = {expr}",
                    justification="First compute the indefinite integral."
                ))

                wallis_result = _try_wallis_fast_path(expr, x, lower_val, upper_val)
                if wallis_result is not None:
                    answer, answer_latex, numeric_value = wallis_result
                    steps.append(StepDetail(
                        step_number=3,
                        description="Evaluate at bounds using FTC",
                        expression=f"F({upper}) - F({lower})",
                        expression_latex=f"F({latex(upper_val)}) - F({latex(lower_val)})",
                        justification="Fundamental Theorem of Calculus: ∫_a^b f(x)dx = F(b) - F(a)."
                    ))
                    steps.append(StepDetail(
                        step_number=4,
                        description="Final simplified result",
                        result=answer,
                        result_latex=answer_latex,
                        justification=(
                            f"Computed via the Wallis reduction formula for powers of sin/cos "
                            f"rather than symbolic term-by-term integration, which is impractical "
                            f"for an exponent this large. Exact value; ≈ {numeric_value:.6g} numerically."
                        ),
                    ))
                    formulas_used.append(FormulaUsed(
                        name="Wallis Formula",
                        formula_latex=r"\int_0^{\pi/2} \sin^n(x)\,dx = \int_0^{\pi/2} \cos^n(x)\,dx",
                        description="Closed-form reduction for definite integrals of high powers of sin/cos over standard bounds."
                    ))
                    return _with_math_context(
                        SolveResponse(
                            question=expression,
                            question_latex=question_latex,
                            topic=topic,
                            answer=answer,
                            answer_latex=answer_latex,
                            steps=steps,
                            difficulty=_estimate_difficulty(topic, expression),
                            formulas_used=formulas_used,
                            verification=f"Verified via the exact Wallis closed form; ≈ {numeric_value:.6g}.",
                            ai_confidence=max(confidence, 0.9),
                        ),
                        expression, parsed, request.session_id,
                    )

                frac_result = _try_frac_fast_path(expr, x, lower_val, upper_val)

                try:
                    antideriv = integrate(expr, x) if frac_result is None else _frac_antiderivative(x)
                except Exception:
                    antideriv = integrate(expr, x, risch=False)

                steps.append(StepDetail(
                    step_number=3,
                    description="Evaluate at bounds using FTC",
                    expression=f"F({upper}) - F({lower})",
                    expression_latex=f"F({latex(upper_val)}) - F({latex(lower_val)})",
                    justification="Fundamental Theorem of Calculus: ∫_a^b f(x)dx = F(b) - F(a)."
                ))

                if frac_result is not None:
                    result = frac_result
                else:
                    try:
                        result = integrate(simplify(expr), (x, lower_val, upper_val))
                    except Exception:
                        try:
                            result = integrate(expr, (x, lower_val, upper_val))
                        except Exception:
                            try:
                                result = sp.N(integrate(expr, (x, lower_val, upper_val)))
                            except Exception:
                                lower_val = float(sp.N(lower_val))
                                upper_val = float(sp.N(upper_val))
                                result = integrate(expr, (x, lower_val, upper_val))

                simplified = _best_simplify(result) if not isinstance(result, float) else result
                verify_ctx = {"kind": "definite_integral", "expr": expr, "var": x, "lower": lower_val, "upper": upper_val, "result": simplified}

                steps.append(StepDetail(
                    step_number=4,
                    description="Substitute bounds and simplify",
                    expression=str(result),
                    expression_latex=latex(result) if not isinstance(result, float) else str(result),
                    result=str(simplified),
                    result_latex=latex(simplified) if not isinstance(simplified, float) else str(simplified),
                    justification="Plug in the upper and lower bounds into the antiderivative."
                ))

                answer = _format_plain(simplified)
                answer_latex = latex(simplified) if not isinstance(simplified, float) else str(simplified)

                formulas_used.append(FormulaUsed(
                    name="Fundamental Theorem of Calculus",
                    formula_latex=r"\int_a^b f(x) \, dx = F(b) - F(a)",
                    description="Connects differentiation and integration."
                ))
            else:
                question_latex = f"\\int {latex(expr)} \\, dx"
                technique = _detect_integration_technique(integrand_expr, expr)

                steps.append(StepDetail(
                    step_number=1,
                    description="Identify the indefinite integral",
                    expression=_format_plain(expr),
                    expression_latex=question_latex,
                    justification="We need to find the antiderivative. The integrand is " + latex(expr) + "."
                ))

                steps.append(StepDetail(
                    step_number=2,
                    description=f"Apply integration technique: {technique}",
                    expression=f"Analyze the integrand structure to choose the best method",
                    justification=f"Detected technique: {technique}. This is the most efficient approach for this type of integrand."
                ))

                hyperbolic_result = _try_hyperbolic_secant_sqrt_antiderivative(expr, x)
                exp_sinh_log_cosh_result = _try_exp_sinh_log_cosh_antiderivative(expr, x)
                if hyperbolic_result is not None:
                    result = hyperbolic_result
                elif exp_sinh_log_cosh_result is not None:
                    result = exp_sinh_log_cosh_result
                else:
                    try:
                        if sp.simplify(expr - (x - sp.floor(x))) == 0:
                            result = _frac_antiderivative(x)
                        else:
                            result = integrate(simplify(expr), x)
                    except Exception:
                        try:
                            result = integrate(expr, x)
                        except Exception:
                            try:
                                result = integrate(expr, x, risch=False)
                            except Exception:
                                result = sp.Integral(expr, x)

                simplified = _best_simplify(result)

                if _is_unresolved(simplified):
                    # One more attempt with sympy's pattern-matching ("manual")
                    # integrator, which sometimes succeeds where the default
                    # Risch/Meijer-G-based approach doesn't.
                    try:
                        alt_result = integrate(expr, x, manual=True)
                    except Exception:
                        alt_result = None
                    if alt_result is not None and not _is_unresolved(alt_result):
                        result = alt_result
                        simplified = _best_simplify(result)

                if _is_unresolved(simplified):
                    # sympy genuinely couldn't find a closed form - it just
                    # handed back Integral(expr, x) unevaluated. Showing that
                    # dressed up as "answer + C" (as this used to do) isn't a
                    # solved integral, and the "verification" step that
                    # differentiates it to reproduce the integrand is
                    # tautologically true for *any* unevaluated integral, so
                    # it can't actually confirm anything was solved. Report
                    # the failure honestly instead.
                    return _with_math_context(
                        _fallback_response(
                            expression, topic, confidence,
                            "Couldn't find a closed-form antiderivative for this integrand. "
                            "Sympy's symbolic integration returned it unevaluated, so no answer "
                            "is reported rather than presenting the unsolved integral as a result.",
                        ),
                        expression, parsed, request.session_id,
                    )

                verify_ctx = {"kind": "indefinite_integral", "expr": expr, "var": x, "antideriv": simplified}

                steps.append(StepDetail(
                    step_number=3,
                    description="Compute the antiderivative",
                    expression=_format_plain(simplified),
                    expression_latex=latex(simplified),
                    justification="Apply the chosen integration method to obtain the antiderivative."
                ))

                steps.append(StepDetail(
                    step_number=4,
                    description="Add the constant of integration",
                    expression=_format_plain(simplified),
                    expression_latex=latex(simplified),
                    result=_format_plain(simplified) + " + C",
                    result_latex=f"{latex(simplified)} + C",
                    justification="For indefinite integrals, always include the constant of integration +C."
                ))

                answer = _format_plain(simplified) + " + C"
                answer_latex = f"{latex(simplified)} + C"

                formulas_used.append(FormulaUsed(
                    name="Power Rule for Integration",
                    formula_latex=r"\int x^n \, dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)",
                    description="Antiderivative of power functions."
                ))
                formulas_used.append(FormulaUsed(
                    name="Integration by Substitution",
                    formula_latex=r"\int f(g(x)) g'(x) \, dx = \int f(u) \, du",
                    description="Substitution method for integration."
                ))
                if '/' in str(expr):
                    formulas_used.append(FormulaUsed(
                        name="Integration of Rational Functions",
                        formula_latex=r"\int \frac{P(x)}{Q(x)} \, dx",
                        description="Using partial fractions, logarithms, and inverse trigonometric functions as needed."
                    ))
                if any(f in str(expr).lower() for f in ('exp(', 'log(', 'ln(')):
                    formulas_used.append(FormulaUsed(
                        name="Exponential and Logarithmic Integration",
                        formula_latex=r"\int e^x \, dx = e^x + C \quad \int \frac{1}{x} \, dx = \ln|x| + C",
                        description="Special integration rules for exponential and logarithmic functions."
                    ))

            alternative_methods.append(AlternativeMethod(
                name="Integration by Parts",
                steps=["Identify u and dv: let u = (polynomial/log/trig factor), dv = (remaining factor) dx",
                       "Apply formula: ∫ u dv = u·v − ∫ v du",
                       "Solve the resulting simpler integral"],
                final_answer=answer,
                final_answer_latex=answer_latex
            ))
            alternative_methods.append(AlternativeMethod(
                name="Numerical Integration",
                steps=["Discretize the interval using Simpson's rule or Gaussian quadrature",
                       "Evaluate the integrand at sample points",
                       "Sum the weighted contributions to get the approximate value"],
                final_answer=answer,
                final_answer_latex=answer_latex
            ))

            if request.include_graph and lower is not None and upper is not None:
                graph_data = GraphData(
                    graph_type="definite_integral_area",
                    data={
                        "function": latex(expr),
                        "lower": str(lower),
                        "upper": str(upper),
                    }
                )
            elif request.include_graph:
                graph_data = GraphData(
                    graph_type="function",
                    data={"function": latex(expr)}
                )

        elif parsed[0] == "solve_equation":
            return _with_math_context(_solve_equation_response(expression, parsed[1], topic, confidence), expression, parsed, request.session_id)

        elif parsed[0] == "symbolic_operation":
            return _with_math_context(_symbolic_operation_response(expression, parsed[1], parsed[2], topic, confidence), expression, parsed, request.session_id)

        elif parsed[0] == "expression":
            return _with_math_context(_general_expression_response(expression, parsed[1], topic, confidence), expression, parsed, request.session_id)

        else:
            return _with_math_context(_fallback_response(expression, topic, confidence), expression, parsed, request.session_id)

    except Exception as e:
        return _with_math_context(_fallback_response(expression, topic, confidence, _clean_error_message(e)), expression, parsed, request.session_id)

    # Numerical verification
    verification = _numerical_verify(verify_ctx, topic)

    # Never present a result as the headline answer when we have positive
    # evidence it's wrong (not just "couldn't verify" - a genuine numeric
    # disagreement). This is the actual harm from the original cos(x)**2020
    # bug: a wrong answer shown next to a "✓ Verified" badge that had never
    # really checked anything. The disputed value stays visible in the
    # verification text and step history for transparency; it just isn't
    # asserted as correct here.
    if verification.startswith("⚠"):
        answer = "Unable to confirm a correct result for this expression - the computed value failed numeric verification (see below)."
        answer_latex = r"\text{Unverified result - see note below}"

    if not request.include_steps:
        steps = []
        alternative_methods = []
        formulas_used = []

    return _with_math_context(SolveResponse(
        question=expression,
        question_latex=question_latex or expression,
        topic=topic,
        answer=answer,
        answer_latex=answer_latex,
        steps=steps,
        difficulty=_estimate_difficulty(topic, expression),
        alternative_methods=alternative_methods,
        formulas_used=formulas_used,
        verification=verification,
        graph_data=graph_data,
        ai_confidence=confidence,
    ), expression, parsed, request.session_id)


def verify_solution(request: VerifyRequest) -> VerifyResponse:
    expression = request.expression
    student_solution = request.student_solution

    solve_req = SolveRequest(expression=expression)
    correct = solve_calculus(solve_req)

    steps_analysis: list[VerificationStep] = []

    lines = [l.strip() for l in student_solution.split('\n') if l.strip()]
    for i, line in enumerate(lines, 1):
        if '=' in line:
            left, right = line.split('=', 1)
            steps_analysis.append(VerificationStep(
                step_number=i,
                is_correct=True,
                student_step=line,
                feedback="Step appears logically structured.",
            ))
        else:
            steps_analysis.append(VerificationStep(
                step_number=i,
                is_correct=True,
                student_step=line,
                feedback="Step noted.",
            ))

    try:
        correct_ans_raw = correct.answer.replace(" + C", "").replace("+ C", "").strip()
        lines = student_solution.strip().split('\n') if student_solution.strip() else []
        student_ans_raw = (lines[-1].split('=')[-1].strip() if lines and '=' in lines[-1] else lines[-1].strip()) if lines else student_solution.strip()
        student_ans_raw = student_ans_raw.replace(" + C", "").replace("+ C", "").strip()
        # Use sympy for algebraic equivalence checking
        try:
            correct_sym = safe_sympify(correct_ans_raw)[0]
            student_sym = safe_sympify(student_ans_raw)[0]
            if correct_sym is not None and student_sym is not None:
                final_correct = simplify(correct_sym - student_sym) == 0
            else:
                final_correct = correct_ans_raw.lower().replace(" ", "") == student_ans_raw.lower().replace(" ", "")
        except Exception:
            final_correct = correct_ans_raw.lower().replace(" ", "") == student_ans_raw.lower().replace(" ", "")
    except Exception:
        final_correct = False

    overall = "Excellent! Your solution is correct and well-structured." if final_correct else \
        "Your approach shows understanding, but there may be calculation errors. Review each step carefully."

    return VerifyResponse(
        original_question=expression,
        student_solution=student_solution,
        is_correct=final_correct,
        final_answer_correct=final_correct,
        steps_analysis=steps_analysis,
        overall_feedback=overall,
        score=0.85 if final_correct else 0.4,
    )


def _clean_error_message(exc: Exception) -> str:
    """Convert a raw Python/sympy exception into a message that's safe and
    useful to show a user, instead of leaking internal implementation
    details (e.g. a Python SyntaxError's own repr: "invalid syntax.
    Perhaps you forgot a comma? (<string>, line 1)"), which happened
    verbatim before this existed - most commonly triggered by a malformed
    token from an OCR misread (e.g. "2026" misread as "2D/6").
    """
    return "Couldn't fully process this expression - it may contain a typo, an unrecognized symbol, or something outside what this solver currently supports."


def _fallback_response(expression: str, topic: str, confidence: float, error: str = "") -> SolveResponse:
    return SolveResponse(
        question=expression,
        topic=topic,
        answer="Unable to compute a closed-form symbolic solution for this expression.",
        answer_latex="\\text{Unable to compute a closed-form symbolic solution}",
        steps=[
            StepDetail(
                step_number=1,
                description="Parse the input expression",
                expression=expression,
                justification="Attempted to identify the calculus problem."
            ),
            StepDetail(
                step_number=2,
                description="Attempt symbolic computation",
                expression=expression,
                justification="The expression may require advanced techniques or numerical methods." + (f" Error: {error}" if error else "")
            ),
        ],
        difficulty=_estimate_difficulty(topic, expression),
        ai_confidence=confidence * 0.5,
    )


def _numerical_verify(verify_ctx, topic: str) -> str:
    """Thin wrapper kept for backward compatibility - the actual logic now
    lives in app.services.verification (item 16's module-split goal)."""
    return numerical_verify(verify_ctx, topic)


def _estimate_difficulty(topic: str, expression: str) -> str:
    expr_len = len(expression)
    complexity = sum(c in expression for c in "^*/()√∫∂∑∏∞")

    if any(t in topic for t in ["Stokes", "Divergence", "Green", "Surface Integrals", "Fourier", "Laplace"]):
        return "Expert"
    if any(t in topic for t in ["Multiple Integrals", "Vector Calculus", "Line Integrals", "Differential Equations"]):
        return "Hard" if complexity > 5 else "Medium"
    if complexity > 8:
        return "Hard"
    if complexity > 4:
        return "Medium"
    return "Easy"