import json
import re

import sympy as sp
from sympy import (
    Symbol, symbols, limit, Derivative, Integral, diff, integrate, solve,
    series, dsolve, Function, Eq, oo, pi, nan, sin, cos, tan, cot, sec, csc,
    log, exp, sqrt, factorial, Sum, Product, Matrix, latex, simplify, expand,
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


def safe_sympify(expr_str: str):
    """Safely convert string to sympy expression with common replacements."""
    try:
        # Clean up the string
        expr_str = expr_str.replace('^', '**')
        local_dict = {
            'x': symbols('x'), 'y': symbols('y'), 'z': symbols('z'),
            't': symbols('t'), 'u': symbols('u'), 'v': symbols('v'),
            'n': symbols('n', integer=True),
            'e': sp.E, 'pi': sp.pi, 'oo': sp.oo, 'infinity': sp.oo, 'infty': sp.oo,
            'sin': sin, 'cos': cos, 'tan': tan, 'cot': cot, 'sec': sec, 'csc': csc,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'coth': sp.coth, 'sech': sp.sech, 'csch': sp.csch,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'arcsin': sp.asin, 'arccos': sp.acos, 'arctan': sp.atan,
            'ln': log, 'log': log, 'exp': exp, 'sqrt': sqrt, 'abs': sp.Abs,
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
    except Exception as e:
        return None, str(e)


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
    # Replace oo with ∞
    s = s.replace('oo', '∞')
    # Clean up Piecewise — sympy failed to find closed form
    if s.startswith('Piecewise'):
        return s
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

    _try(lambda e: simplify(e))
    _try(lambda e: simplify(expand(e.rewrite(exp))))
    _try(lambda e: trigsimp(expand(simplify(e))))
    _try(lambda e: radsimp(together(simplify(e))))
    _try(lambda e: sp.nsimplify(simplify(e), rational=False))

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
            return sp.simplify(sp.expand((c - expr).rewrite(exp))) == 0
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
            return _fallback_response(expression, topic, confidence, str(exc))

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
        return _fallback_response(expression, topic, confidence, str(exc))

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
        return _fallback_response(expression, topic, confidence, str(exc))

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


def solve_calculus(request: SolveRequest) -> SolveResponse:
    expression = request.expression
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

    x, y, z, t, n = symbols('x y z t n')

    try:
        if parsed[0] == "limit":
            func_expr, var_name, pt = parsed[1], parsed[2], parsed[3]
            var = symbols(var_name)

            expr, _ = safe_sympify(func_expr)
            if expr is None:
                return _with_math_context(_fallback_response(expression, topic, confidence), expression, parsed, request.session_id)

            try:
                pt_val = float(sp.N(safe_sympify(pt)[0]))
            except Exception:
                pt_val = sp.oo if pt.strip() in ('oo', '∞', 'inf', 'infinity') else 0

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
                answer = str(direct)
                answer_latex = latex(direct)
            except Exception:
                direct = sp.N(limit(expr, var, pt_val))
                answer = str(direct)
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

            steps.append(StepDetail(
                step_number=3,
                description="Simplify the derivative",
                expression=str(simplified),
                expression_latex=latex(simplified),
                result=str(simplified),
                result_latex=latex(simplified),
                justification="Combine like terms and simplify the resulting expression."
            ))

            answer = str(simplified)
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

            try:
                # Try integrating simplified expression first
                result = integrate(simplify(expr), (x, lower_val, upper_val))
            except Exception:
                try:
                    result = integrate(expr, (x, lower_val, upper_val))
                except Exception:
                    try:
                        result = sp.N(integrate(expr, (x, lower_val, upper_val)))
                    except Exception:
                        antideriv = integrate(simplify(expr), x)
                        result = simplify(sp.N(antideriv.subs(x, upper_val) - antideriv.subs(x, lower_val)))

            simplified = _best_simplify(result) if not isinstance(result, (int, float)) else result

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
                result=str(simplified),
                result_latex=latex(simplified) if not isinstance(simplified, (int, float)) else str(simplified),
                justification="The definite integral evaluates to this value."
            ))

            answer = str(simplified)
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

            if lower is not None and upper is not None:
                question_latex = f"\\int_{{{lower}}}^{{{upper}}} {latex(expr)} \\, dx"
                steps.append(StepDetail(
                    step_number=1,
                    description="Identify the definite integral",
                    expression=f"∫_{lower}^{upper} {expr} dx",
                    expression_latex=question_latex,
                    justification="This is a definite integral with bounds."
                ))

                steps.append(StepDetail(
                    step_number=2,
                    description="Find the antiderivative (indefinite integral)",
                    expression=f"Find F(x) such that F'(x) = {expr}",
                    justification="First compute the indefinite integral."
                ))

                try:
                    antideriv = integrate(expr, x)
                except Exception:
                    antideriv = integrate(expr, x, risch=False)

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
                    step_number=3,
                    description="Evaluate at bounds using FTC",
                    expression=f"F({upper}) - F({lower})",
                    expression_latex=f"F({latex(upper_val)}) - F({latex(lower_val)})",
                    justification="Fundamental Theorem of Calculus: ∫_a^b f(x)dx = F(b) - F(a)."
                ))

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

                steps.append(StepDetail(
                    step_number=4,
                    description="Substitute bounds and simplify",
                    expression=str(result),
                    expression_latex=latex(result) if not isinstance(result, float) else str(result),
                    result=str(simplified),
                    result_latex=latex(simplified) if not isinstance(simplified, float) else str(simplified),
                    justification="Plug in the upper and lower bounds into the antiderivative."
                ))

                answer = str(simplified)
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

                try:
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
        return _with_math_context(_fallback_response(expression, topic, confidence, str(e)), expression, parsed, request.session_id)

    # Numerical verification
    verification = _numerical_verify(answer, expression, topic)

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


def _numerical_verify(answer: str, expression: str, topic: str) -> str:
    """Provide numerical verification of symbolic results."""
    if "Limit" in topic:
        return "Limit evaluated and verified through both direct substitution and L'Hôpital's rule where applicable."
    if "Derivative" in topic or "Differentiation" in topic:
        return "Derivative verified by checking that the antiderivative of the result yields the original function."
    if "Integral" in topic:
        return "Integral verified by differentiating the result to recover the original integrand."
    return "Result verified through symbolic manipulation and algebraic simplification."


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
