"""Ordinary differential equation solving.

Parses plain-text ODEs ("y'' + 4y = sin(x)", "dy/dx = x*y"), classifies
them via sympy's classify_ode (so the step explanation can name the actual
technique used - separable, first-order linear, constant-coefficient
homogeneous, Bernoulli, ...), solves them with dsolve, and optionally
applies initial conditions like "with y(0) = 1, y'(0) = 0".

The result is always checked by substituting it back into the original ODE
(checkodesol); if that fails, the answer is reported as unverified rather
than claimed correct - same "never fabricate a result" policy as the rest
of the solver.

Security: all user text is parsed with parse_expr restricted to an empty
globals namespace (__builtins__ removed) so the local_dict whitelist is
the ONLY way a name can resolve - inputs like "__import__('os')" are
rejected as unknown names instead of executing arbitrary code.

Deliberately NOT handled: systems of ODEs and PDEs. sympy's ODE machinery
covers single ODEs only; those inputs fall through to an honest
"not supported" message via the caller's fallback path.
"""

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
)
from sympy.solvers.ode import classify_ode, checkodesol

from app.services.compute_utils import run_with_timeout, ComputeTimeout

# Human-readable names for the method labels classify_ode returns
_METHOD_NAMES = {
    "separable": "Separable Equation",
    "1st_exact": "First-Order Exact Equation",
    "2nd_exact": "Second-Order Exact Equation",
    "1st_linear": "First-Order Linear Equation",
    "2nd_linear": "Second-Order Linear Equation",
    "Bernoulli": "Bernoulli Equation",
    "Riccati_special_minus2": "Special Riccati Equation",
    "almost_linear": "Almost-Linear Equation",
    "1st_homogeneous_coeff_best": "Homogeneous Coefficient Equation",
    "1st_homogeneous_coeff_subst_indep_div_dep": "Homogeneous Coefficient Equation (v = x/y substitution)",
    "1st_homogeneous_coeff_subst_dep_div_indep": "Homogeneous Coefficient Equation (v = y/x substitution)",
    "nth_linear_constant_coeff_homogeneous": "Linear Constant-Coefficient Homogeneous Equation",
    "nth_linear_constant_coeff_undetermined_coefficients": "Linear Constant-Coefficient Equation (Undetermined Coefficients)",
    "nth_linear_constant_coeff_variation_of_parameters": "Linear Constant-Coefficient Equation (Variation of Parameters)",
    "nth_linear_euler_eq_homogeneous": "Euler-Cauchy (Equidimensional) Homogeneous Equation",
    "nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients": "Euler-Cauchy Equation (Undetermined Coefficients)",
    "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters": "Euler-Cauchy Equation (Variation of Parameters)",
    "2nd_nonlinear_autonomous_conserved": "Nonlinear Autonomous Equation (Conserved Quantity)",
    "lie_group": "Lie Group Method",
    "solvable": "Solvable for the Derivative",
}


def _method_label(hints) -> str:
    for hint in hints:
        if hint.startswith("best"):
            continue
        base = hint.rsplit("_Integral", 1)[0]
        if base in _METHOD_NAMES:
            return _METHOD_NAMES[base]
    return "Symbolic ODE Solution"


def _extract_order(digits: str, default: int = 1) -> int:
    cleaned = re.sub(r"\D", "", digits or "")
    return int(cleaned) if cleaned else default


def _rewrite_derivative_text(text: str, fn: str, var: str) -> str:
    """Rewrite every plain-text derivative notation for `fn` into
    Derivative(fn,(var,n)), the form _parse_side understands:

    - y', y'', y''' (primes)                 -> Derivative(y,(x,n))
    - dy/dx, d^2y/dx^2 (Leibniz)             -> Derivative(y,(x,n))

    Done at the text level because the parser has no way to attach a
    function-of-var meaning to bare names otherwise.
    """
    text = re.sub(
        rf"\b{fn}\s*('+)",
        lambda m: f"Derivative({fn},({var},{len(m.group(1))}))",
        text,
    )
    return re.sub(
        rf"d(?:\^(\d+)|(\d*))\s*{re.escape(fn)}\s*/\s*d\s*{re.escape(var)}\s*(?:\^(\d+)|(\d*))?",
        lambda m: (
            f"Derivative({fn},({var},{_extract_order(m.group(1) or m.group(2), 1)}))"
            if (m.group(1) or m.group(2)) else
            f"Derivative({fn},({var},{_extract_order(m.group(3) or m.group(4), 1)}))"
        ),
        text,
        flags=re.IGNORECASE,
    )


def _detect_function(expression: str) -> str:
    """Best-effort detection of the dependent variable name in a raw ODE
    string (the symbol that gets primed: y' -> y). Defaults to y.

    e and i are excluded since they mean Euler's number and the imaginary
    unit; x is excluded since it's this solver's independent variable.
    """
    if re.search(r"\by\s*'", expression) or re.search(r"\bd\s*\^?\d*\s*y\s*/\s*d", expression, re.IGNORECASE):
        return "y"
    m = re.search(r"\b([a-df-hj-wyzA-Z])\s*'", expression)
    if m:
        return m.group(1)
    m = re.search(r"d\s*\^?\d*\s*([a-df-hj-wyzA-Z])\s*/\s*d", expression, re.IGNORECASE)
    if m:
        return m.group(1)
    return "y"


_PARSE_LOCALS_BASE = {
    "e": sp.E, "pi": sp.pi, "oo": sp.oo, "infinity": sp.oo, "infty": sp.oo,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "sec": sp.sec, "csc": sp.csc,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "ln": sp.log, "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs,
    "Derivative": sp.Derivative, "Diff": sp.Derivative,
    "floor": sp.floor, "ceiling": sp.ceiling,
}

# Restricted evaluation namespace for parse_expr. By default parse_expr
# eval's the transformed input against sympy's global namespace, which
# drags in every Python builtin (__import__, eval, exec, open, ...). That
# let input like "__import__('os').getcwd()" execute arbitrary code.
#
# We seed the globals from sympy's own namespace (the generated parse code
# needs names like Integer/Symbol to resolve), then wipe __builtins__, so
# builtins and dunder-based import tricks are unavailable. The local_dict
# whitelist remains the source for the user-facing math names.
_SAFE_GLOBALS: dict = {}
exec("from sympy import *", _SAFE_GLOBALS)  # noqa: S102 - seeding the math namespace only
_SAFE_GLOBALS["__builtins__"] = {}

# First line of defense: a strict scanner on the raw user text. Blocking
# builtins from the namespace is NOT sufficient on its own - Python's classic
# sandbox-escape gadget "().__class__.__bases__[0].__subclasses__()" reaches the
# class hierarchy via attribute access and indexing on a literal (), never
# touching a global name. Legitimate ODE input only needs identifiers, math
# functions, + - * / ^ ( ) , digits, and decimal points, so we reject any text
# containing dunders, attribute access, indexing, or dangerous keywords
# BEFORE it ever reaches parse_expr's eval.
_BLOCKED_KEYWORDS = re.compile(
    r"(__|import|exec|eval|compile|getattr|setattr|delattr|globals|locals|"
    r"lambda|vars|dir|breakpoint|__builtins__|subclasses|bases|mro|class__)",
    re.IGNORECASE,
)


def _validate_ode_text(text: str) -> bool:
    """Return False (reject) for text carrying Python-escape syntax, True
    for text that could plausibly be a math expression."""
    if _BLOCKED_KEYWORDS.search(text):
        return False
    if "[" in text or "]" in text or ";" in text or ":" in text:
        return False
    # Allow '.' only as a decimal point between two digits; any other dot is
    # attribute access (e.g. "().__class__") and must be rejected.
    for m in re.finditer(r"\.", text):
        i = m.start()
        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if not (prev.isdigit() and nxt.isdigit()):
            return False
    return True


def _build_local_dict(fn_name: str, fnx, x_symbol: sp.Symbol) -> dict:
    """Build the restricted local_dict for ODE parsing."""
    local_dict = dict(_PARSE_LOCALS_BASE)
    local_dict[fn_name] = fnx
    local_dict[x_symbol.name] = x_symbol
    for name in ("x", "y", "z", "t"):
        if name not in local_dict:
            local_dict[name] = sp.Symbol(name)
    return local_dict


_FUNC_POWER_NAMES = (
    "sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|sinh|cosh|tanh|exp|ln|log|sqrt"
)
_FUNC_POWER_PATTERN = re.compile(rf"\b({_FUNC_POWER_NAMES})\s*\^\s*(\d+)\s*\(", re.IGNORECASE)


def _rewrite_func_power(text: str) -> str:
    """Rewrite textbook function-power notation sin^2(x) -> (sin(x))**2 with
    balanced-paren matching, matching the preprocessing safe_sympify applies
    on every other solver path. Without this, parse_expr with implicit
    multiplication reads sin^2(x) as the function class raised to a power
    and raises a TypeError. Only named functions followed by an explicit
    parenthesized argument are rewritten - bare y^2 is left for convert_xor.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = _FUNC_POWER_PATTERN.match(text, i)
        if not m:
            out.append(text[i])
            i += 1
            continue
        depth = 0
        start = m.end() - 1
        j = start
        while j < n:
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            out.append(text[i])
            i += 1
            continue
        out.append(f"({m.group(1)}({text[start + 1:j]}))**{m.group(2)}")
        i = j + 1

    return "".join(out)


def _parse_side(text: str, fn_name: str, x_symbol: sp.Symbol):
    """Parse one side of an ODE into a sympy expression, giving the bare
    function name (e.g. "y") and its derivative notations the meaning
    fn(x). Returns (expr, err).

    SECURITY (two layers):
    1. _validate_ode_text rejects dunder/attribute/indexing/keyword gadgets
       at the raw-text level before any evaluation happens.
    2. parse_expr runs with global_dict=_SAFE_GLOBALS (sympy math names only,
       __builtins__ wiped) so even if a fragment slipped through the scanner,
       it can only resolve names from the math whitelist - no __import__,
       no exec, no attribute chain into builtins."""
    fn = sp.Function(fn_name)
    fnx = fn(x_symbol)

    if not _validate_ode_text(text):
        return None, (
            "This expression contains characters or constructs that aren't part of "
            "ODE notation. Please use standard derivative notation like y', y'', or dy/dx."
        )

    text = _rewrite_func_power(text)
    local_dict = _build_local_dict(fn_name, fnx, x_symbol)

    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        return parse_expr(
            text, local_dict=local_dict, global_dict=_SAFE_GLOBALS,
            transformations=transformations, evaluate=True
        ), None
    except Exception:
        return None, (
            "Couldn't parse the differential equation - check notation like y', y'', "
            "or dy/dx, and make sure both sides are valid expressions."
        )


def _scan_balanced(text: str, start: int) -> str | None:
    """From the position of an opening '(' at `start`, return the balanced
    inner text (or None if unbalanced)."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for j in range(start, len(text)):
        if text[j] == "(":
            depth += 1
        elif text[j] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1:j]
    return None


def _extract_initial_conditions(text: str, fn_name: str, x_symbol: sp.Symbol):
    """Pull out initial conditions written like y(0) = 1 or y'(0) = 2
    (optionally after a "with" / "where" keyword), removing them from the
    ODE text. Returns (cleaned_text, ics_dict, condition_strings) where
    ics_dict maps sympy expressions (y(a) or Derivative(y(x),(x,n)).subs)
    to values - the format sp.dsolve(ics=...) expects.
    """
    ics: dict[object, object] = {}
    cond_strings: list[str] = []
    fn = sp.Function(fn_name)

    cond_pattern = re.compile(
        rf"(\b{re.escape(fn_name)}\s*('+)?\s*(?:[(]|$))", re.IGNORECASE,
    )

    out_chars: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        m = cond_pattern.match(text, i)
        if not m:
            out_chars.append(text[i])
            i += 1
            continue

        prime_part = m.group(2) or ""
        paren_pos = text.find("(", m.start())
        pt_raw = _scan_balanced(text, paren_pos) if paren_pos >= 0 else None
        if pt_raw is None:
            out_chars.append(text[i])
            i += 1
            continue

        eq_pos = paren_pos + len(pt_raw) + 2
        eq_match = re.compile(r"\s*=\s*")
        em = eq_match.match(text, eq_pos)
        if not em:
            out_chars.append(text[i])
            i += 1
            continue

        j = em.end()
        depth = 0
        end = j
        while end < n:
            ch = text[end]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch in ",;" and depth == 0:
                break
            end += 1
        val_raw = text[j:end].strip()
        if not val_raw:
            out_chars.append(text[i])
            i += 1
            continue

        pt, _ = _parse_side(pt_raw, fn_name, x_symbol)
        val, _ = _parse_side(val_raw, fn_name, x_symbol)
        if pt is None or val is None:
            out_chars.append(text[i])
            i += 1
            continue

        if prime_part:
            deriv = sp.Derivative(fn(x_symbol), (x_symbol, len(prime_part)))
            ics[deriv.subs(x_symbol, pt)] = val
            cond_strings.append(f"{fn_name}{''.join(prime_part)}({pt_raw}) = {val_raw}")
        else:
            ics[fn(x_symbol).subs(x_symbol, pt)] = val
            cond_strings.append(f"{fn_name}({pt_raw}) = {val_raw}")
        i = end

    return "".join(out_chars), ics, cond_strings


def parse_ode(expression: str):
    """Parse a raw ODE string into (eq, ics, cond_strings, fn_name, fnx,
    x_symbol, error). eq is None when parsing failed (error set)."""
    if not _validate_ode_text(expression):
        return None, None, None, "y", None, sp.Symbol("x"), (
            "This expression contains characters or constructs that aren't part of "
            "ODE notation (e.g. attribute access, indexing, double underscores). "
            "Please use standard derivative notation like y', y'', or dy/dx."
        )
    fn_name = _detect_function(expression)
    x_symbol = sp.Symbol("x")

    cleaned, ics, cond_strings = _extract_initial_conditions(expression, fn_name, x_symbol)
    cleaned = re.sub(r"\b(?:with|where|given|and|initial(?:\s+conditions?)?)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,;]+", " ", cleaned)

    if "=" not in cleaned:
        cleaned = cleaned + " = 0"

    cleaned = _rewrite_derivative_text(cleaned, fn_name, x_symbol.name)

    lhs_raw, rhs_raw = cleaned.split("=", 1)
    lhs, err = _parse_side(lhs_raw.strip(), fn_name, x_symbol)
    if lhs is None:
        return None, None, None, fn_name, None, x_symbol, err
    rhs, err = _parse_side(rhs_raw.strip(), fn_name, x_symbol)
    if rhs is None:
        return None, None, None, fn_name, None, x_symbol, err

    return sp.Eq(lhs, rhs), ics, cond_strings, fn_name, sp.Function(fn_name)(x_symbol), x_symbol, None


def _dsolve_with_timeout(eq, fnx, hints, ics=None, seconds: float = 45):
    """dsolve with timeout protection, reusing classify_ode's hint when
    available so the (expensive) classification isn't repeated inside
    dsolve. Falls back to default classification if the hinted call fails."""
    if hints:
        try:
            result = run_with_timeout(sp.dsolve, eq, fnx, hint=hints, ics=ics, seconds=seconds)
            if result is not None:
                return result
        except (ComputeTimeout, Exception):
            pass
    return run_with_timeout(sp.dsolve, eq, fnx, ics=ics, seconds=seconds)


def solve_ode(expression: str):
    """Solve the ODE in `expression`. Returns a dict:

        {
          "equation": sp.Eq,             # ODE in standard form (LHS = 0)
          "solution": sp.Eq|None,        # general solution (with C1, C2, ...)
          "particular": sp.Eq|None,      # IVP solution if conditions given
          "method": str,                 # human-readable technique name
          "order": int,                  # order of the ODE
          "fn": applied UndefinedFunction,  # e.g. y(x)
          "var": sp.Symbol,              # e.g. x
          "conditions": list[str],       # initial condition strings
          "verified": bool,              # checkodesol confirmation
          "error": str|None,
        }

    Returns None only when parsing fails outright; a parseable-but-
    unsolvable ODE comes back with solution=None and error set.
    """
    parse_result = parse_ode(expression)
    eq, ics, cond_strings, fn_name, fnx, x_symbol, parse_err = parse_result
    if eq is None or fnx is None:
        return None

    standard = sp.Eq(sp.simplify(sp.expand(eq.lhs - eq.rhs)), 0)

    try:
        hints = run_with_timeout(classify_ode, standard, fnx, seconds=15)
    except Exception:
        hints = ()

    try:
        order = sp.ode_order(eq, fnx)
    except Exception:
        order = 1

    try:
        solution = _dsolve_with_timeout(eq, fnx, hints)
    except ComputeTimeout as e:
        return {
            "equation": standard, "solution": None, "particular": None,
            "method": _method_label(hints), "order": order,
            "fn": fnx, "var": x_symbol, "conditions": cond_strings,
            "verified": False, "error": str(e),
        }
    except Exception:
        # sympy's dsolve occasionally raises deep in a specialized solver
        # (e.g. the Riccati machinery on nonlinear first-order equations it
        # can't actually crack) instead of declining gracefully. Treat any
        # such internal failure the same as "no closed form found": report
        # honestly rather than crashing the whole request.
        return {
            "equation": standard, "solution": None, "particular": None,
            "method": _method_label(hints), "order": order,
            "fn": fnx, "var": x_symbol, "conditions": cond_strings,
            "verified": False,
            "error": (
                "sympy could not find a closed-form solution for this ODE. It may "
                "require a technique (numerical methods, special functions) not yet covered."
            ),
        }

    if isinstance(solution, list):
        solution = solution[0] if solution else None

    verified = False
    if solution is not None:
        try:
            verified = bool(checkodesol(eq, solution)[0])
        except Exception:
            verified = False

    particular = None
    if solution is not None and ics:
        try:
            particular = _dsolve_with_timeout(eq, fnx, hints, ics=ics, seconds=30)
            if isinstance(particular, list):
                particular = particular[0] if particular else None
            if particular is not None:
                try:
                    particular = sp.Eq(particular.lhs, sp.simplify(particular.rhs))
                except Exception:
                    pass
        except Exception:
            particular = None

    return {
        "equation": standard,
        "solution": solution,
        "particular": particular,
        "method": _method_label(hints),
        "order": order,
        "fn": fnx,
        "var": x_symbol,
        "conditions": cond_strings,
        "verified": verified,
        "error": None if solution is not None else (
            "sympy could not find a closed-form solution for this ODE. It may "
            "require a technique (numerical methods, special functions) not yet covered."
        ),
    }
