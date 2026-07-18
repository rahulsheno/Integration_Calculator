"""Expression normalization.

Takes raw user/OCR text and rewrites it into a canonical plain-text form that
solver.py's parser can handle reliably: unicode math symbols, superscript
exponents, function-power notation (sin^2(x)), absolute value pipes, and
fractional-part braces are all resolved here, before anything is handed to
sympy. Keeping this separate from solver.py is a first step toward splitting
the monolithic solver module into focused pieces (parsing/normalization,
integration, differentiation, etc.) - see items 2 and 16 of the improvement
spec.

This module only ever produces plain-text math (e.g. "Abs(x)", "frac(x)"),
never sympy objects - actual symbolic evaluation still happens in solver.py.
"""

import re

_SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁻": "-",
}

_UNICODE_REPLACEMENTS = {
    "π": "pi",
    "∞": "oo",
    "×": "*",
    "÷": "/",
    "−": "-",  # unicode minus sign, distinct from hyphen
    "·": "*",
}

# Function names that legitimately take a superscript power, e.g. sin²(x) ->
# sin(x)**2 rather than sin(2)(x) or similar nonsense.
_POWERABLE_FUNCTIONS = (
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "arcsin", "arccos", "arctan",
    "ln", "log",
)


def _convert_superscripts(text: str) -> str:
    """Convert runs of unicode superscript digits to a plain ^n exponent,
    e.g. "x²" -> "x^2", "cos⁵(x)" -> "cos^5(x)"."""
    def repl(match: "re.Match[str]") -> str:
        digits = "".join(_SUPERSCRIPT_MAP[ch] for ch in match.group(0))
        return f"^{digits}"

    return re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", repl, text)


def _rewrite_function_powers(text: str) -> str:
    """Rewrite "sin^2(x)" / "sin²(x)" (already converted to sin^2(x) by this
    point) into "sin(x)^2", which is what the rest of the pipeline expects.
    Left alone, "sin^2(x)" gets misread as sin applied to "2(x)" - i.e. sin
    evaluated at 2*x - a completely different (and wrong) function.
    """
    pattern = r"\b(" + "|".join(_POWERABLE_FUNCTIONS) + r")\^(-?\d+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"

    def repl(match: "re.Match[str]") -> str:
        name, power, arg = match.group(1), match.group(2), match.group(3)
        return f"{name}({arg})^{power}"

    # Apply repeatedly in case of nested/adjacent occurrences
    prev = None
    while prev != text:
        prev = text
        text = re.sub(pattern, repl, text)
    return text


def _rewrite_absolute_value(text: str) -> str:
    """Rewrite matched pairs of "|...|" into Abs(...), e.g. |x| -> Abs(x) and
    |x+1| -> Abs(x+1).

    Nested absolute values (e.g. |x+|x||) are deliberately NOT handled here:
    since "|" has no distinct open/closing glyph, naive left-to-right pairing
    of pipe characters mismatches nested bars and silently produces wrong
    math (e.g. |x+|x|| would turn into the nonsensical "Abs(x+)xAbs()"
    rather than the intended Abs(x + Abs(x))). Correctly resolving nested
    bars needs a real recursive-descent matcher - a separate, larger piece
    of work - so for now, any text with more than one pipe pair is left
    untouched, letting the parser fail cleanly on the stray "|" rather than
    return an incorrect result.
    """
    if text.count("|") == 2:
        match = re.search(r"\|([^|]*)\|", text)
        if match:
            inner = match.group(1)
            text = text[: match.start()] + f"Abs({inner})" + text[match.end():]
    return text


def _rewrite_fractional_part(text: str) -> str:
    """Rewrite standalone "{...}" fractional-part notation into frac(...),
    i.e. {x} -> frac(x) (frac(x) = x - floor(x)). Only applied when the text
    has no LaTeX \\frac/\\left/\\right commands, since in real LaTeX input
    braces are grouping syntax, not the fractional-part operator - applying
    this rewrite there would corrupt e.g. \\frac{1}{2}.
    """
    if "\\" in text:
        return text
    if "{" not in text:
        return text

    while True:
        match = re.search(r"\{([^{}]*)\}", text)
        if not match:
            break
        inner = match.group(1)
        text = text[: match.start()] + f"frac({inner})" + text[match.end():]
    return text


def check_nesting_depth(text: str, max_depth: int = 60) -> str | None:
    """Return an honest error message if `text` has pathologically deep
    bracket nesting (parens, braces, or absolute-value pipes), or None if
    it's within a reasonable depth.

    Item 6 in the improvement spec (things like |x+|x+|x+...|| nested 2026
    levels deep) is explicitly out of scope for this pass - a real fix needs
    a dedicated recursive-pattern representation, not just parsing deeper.
    Rather than let such input hit Python's recursion limit somewhere deep
    in sympy's parser (an unhandled crash), this check catches it up front
    and returns a clear, honest "not supported" message instead - matching
    the spec's own principle of never fabricating a result and never
    crashing on unsupported input.
    """
    depth = 0
    max_seen = 0
    pipe_open = False
    for ch in text:
        if ch in "([{":
            depth += 1
            max_seen = max(max_seen, depth)
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == "|":
            # Pipes can't be tracked as reliably as brackets (no distinct
            # open/close glyph), but a long run of alternating pipes still
            # signals deep nesting worth flagging.
            pipe_open = not pipe_open
            if pipe_open:
                max_seen = max(max_seen, text.count("|") // 2)

    if max_seen > max_depth:
        return (
            f"This expression is nested {max_seen} levels deep, which is far beyond what this solver "
            f"can reliably process. Deeply recursive expressions (like absolute values nested thousands "
            f"of levels) need a dedicated symbolic representation rather than direct expansion - support "
            f"for that isn't implemented yet."
        )
    return None


def _rewrite_sum_product_symbols(text: str) -> str:
    """Rewrite unicode Σ (sum) and Π (product) with subscript/superscript
    bounds into the plain-text "sum ... from n=a to b" form the parser
    already understands, e.g. "Σ(n=1 to oo) 1/n^2" -> "sum 1/n^2 from n=1 to oo".
    Only handles the common "Σ(var=lower to upper) term" and bare "Σ term"
    (bounds supplied separately elsewhere) shapes - arbitrary placement of
    unicode bounds (true subscript/superscript glyphs from OCR) is handled
    upstream by math_ocr.py's LaTeX conversion instead.
    """
    def repl(match: "re.Match[str]") -> str:
        kind = "sum" if match.group(1) == "Σ" else "product"
        var, lower, upper, term = match.group(2), match.group(3), match.group(4), match.group(5)
        return f"{kind} {term.strip()} from {var}={lower.strip()} to {upper.strip()}"

    pattern = r"(Σ|Π)\(\s*([a-zA-Z])\s*=\s*(.+?)\s+to\s+(.+?)\)\s*(.+)"
    return re.sub(pattern, repl, text)


def normalize_expression(text: str) -> str:
    """Run the full normalization pipeline on raw input text. Idempotent -
    calling this twice on already-normalized text is a no-op.
    """
    if not text:
        return text

    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)

    text = _convert_superscripts(text)
    text = _rewrite_sum_product_symbols(text)
    text = _rewrite_function_powers(text)
    text = _rewrite_absolute_value(text)
    text = _rewrite_fractional_part(text)

    # Collapse whitespace introduced/left behind by the rewrites above
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text
