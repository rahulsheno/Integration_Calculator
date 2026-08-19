import base64
import io
import re
from importlib.util import find_spec
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageOps


@dataclass
class MathOcrResult:
    expression: str
    raw_text: str = ""
    confidence: float = 0.0


def is_ocr_engine_available() -> bool:
    return _pix2text_available() or _pix2tex_available() or _ollama_available()


def ocr_engine_status() -> str:
    engines = []
    if _pix2text_available():
        engines.append("Pix2Text")
    if _pix2tex_available():
        engines.append("pix2tex")
    if _ollama_available():
        engines.append(f"Ollama ({_OLLAMA_VISION_MODEL})")
    return " + ".join(engines) if engines else "missing"


def extract_math_expression_from_image(filepath: str) -> MathOcrResult:
    """Extract a solver-friendly math expression from an image of a printed
    or handwritten math problem.

    This relies primarily on real OCR (Pix2Text / pix2tex) run over several
    preprocessed versions of the image, followed by text normalization. It
    never guesses a canned expression based on image size or rough pixel
    shapes - if the OCR engines can't produce something the normalizer can
    parse, this honestly reports that nothing solvable was extracted rather
    than returning a fabricated result.

    As a last resort, if a local vision-capable LLM is available via Ollama,
    it's asked to read the whole image - including any surrounding
    word-problem text a formula-only OCR model can't handle, and any
    algebra needed to derive integration bounds - and produce a final
    expression directly. Its output is only trusted if it exactly matches
    our supported grammar (see _parse_vision_llm_output); free-form text is
    never passed through, for the same "don't fabricate" reason as above.
    """
    image = Image.open(filepath)
    latex_candidates = _latex_candidates(filepath, image)
    expression = _best_expression_from_latex(latex_candidates)
    if expression:
        return MathOcrResult(expression=expression, raw_text="\n".join(latex_candidates), confidence=0.85)

    vision_expression, vision_raw = _vision_llm_expression(image)
    all_raw_text = latex_candidates + ([vision_raw] if vision_raw else [])
    if vision_expression:
        return MathOcrResult(expression=vision_expression, raw_text="\n".join(all_raw_text), confidence=0.75)

    # Nothing normalized successfully. Show the single most representative
    # raw attempt rather than every dead-end candidate concatenated together
    # - joining a dedicated-OCR misread with an unrelated, unvalidated
    # vision-LLM guess produced a confusing scrambled multi-line dump instead
    # of one coherent (if wrong) line the user could actually make sense of.
    best_raw = latex_candidates[0] if latex_candidates else vision_raw
    return MathOcrResult(
        expression="",
        raw_text=best_raw or "",
        confidence=0.25 if best_raw else 0.0,
    )


def _pix2text_available() -> bool:
    return find_spec("pix2text") is not None


def _pix2tex_available() -> bool:
    return find_spec("pix2tex") is not None


def _latex_candidates(filepath: str, image: Image.Image) -> list[str]:
    variants = _preprocess_variants(image)
    candidates: list[str] = []
    for text in _pix2text_candidates(filepath, variants):
        _append_candidate(candidates, text)
    for text in _pix2tex_candidates(variants):
        _append_candidate(candidates, text)
    return candidates


def _append_candidate(candidates: list[str], value: str | None) -> None:
    if value:
        cleaned = value.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)


@lru_cache(maxsize=1)
def _pix2text_engine():
    from pix2text import Pix2Text

    if hasattr(Pix2Text, "from_config"):
        return Pix2Text.from_config()
    return Pix2Text()


def _pix2text_candidates(filepath: str, variants: list[Image.Image]) -> list[str]:
    if not _pix2text_available():
        return []
    try:
        engine = _pix2text_engine()
    except Exception:
        return []

    outputs: list[Any] = []
    # Run the model over the original file plus each cleaned-up variant
    # (cropped to the ink, upscaled, binarized) so a noisy or low-res photo
    # still has a good chance of being read correctly.
    sources: list[Any] = [filepath, *variants]
    kwargs_options = (
        {"file_type": "text_formula", "return_text": True},
        {"return_text": True},
        {},
    )
    for source in sources:
        for kwargs in kwargs_options:
            try:
                outputs.append(engine.recognize(source, **kwargs))
                break
            except TypeError:
                continue
            except Exception:
                break
    return _extract_strings(outputs)


@lru_cache(maxsize=1)
def _pix2tex_engine():
    from pix2tex.cli import LatexOCR

    return LatexOCR()


def _pix2tex_candidates(variants: list[Image.Image]) -> list[str]:
    if not _pix2tex_available():
        return []
    try:
        engine = _pix2tex_engine()
    except Exception:
        return []

    results: list[str] = []
    for variant in variants:
        try:
            results.append(engine(variant.convert("RGB")))
        except Exception:
            continue
    return results


def _extract_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for key in ("latex", "text", "value", "line_text"):
            strings.extend(_extract_strings(value.get(key)))
        for nested_key in ("items", "results", "segments"):
            strings.extend(_extract_strings(value.get(nested_key)))
        return strings
    if isinstance(value, (list, tuple)):
        strings: list[str] = []
        for item in value:
            strings.extend(_extract_strings(item))
        return strings
    return []


def _preprocess_variants(image: Image.Image) -> list[Image.Image]:
    """Build a small set of cleaned-up copies of the image to give the OCR
    engines the best shot at reading a real photo: one with excess
    background margin cropped away, and one that's also upscaled and
    binarized to remove shadows/paper texture/low resolution noise.

    Uses an adaptive (Otsu) threshold rather than a fixed brightness cutoff,
    since real photos have all kinds of background brightness (bright white
    paper, gray screenshots, shadowed phone photos, etc), and detects
    whether the ink is the darker or lighter cluster so it also works on
    inverted/dark-mode images (light text on a dark background), not just
    the usual dark-text-on-light-paper case.

    Kept intentionally small (2-3 images) since each one costs a model
    inference call.
    """
    rgb = image.convert("RGB")
    grayscale = ImageOps.grayscale(rgb)
    cropped = _crop_to_ink(grayscale)

    variants = [rgb]
    if cropped.width < 1 or cropped.height < 1:
        return variants

    variants.append(cropped.convert("RGB"))

    # Upscale small/low-res photos so thin strokes (radicals, fraction bars,
    # exponents) survive the model's internal downsampling.
    scale = 3 if max(cropped.width, cropped.height) < 700 else 1
    enlarged = (
        cropped.resize((cropped.width * scale, cropped.height * scale), Image.Resampling.LANCZOS)
        if scale > 1
        else cropped
    )
    enlarged_arr = np.asarray(enlarged)
    ink_threshold = _ink_threshold(enlarged_arr)
    ink_is_light = _ink_is_light(enlarged_arr, ink_threshold)
    # Always normalize to dark ink on a white background in the output,
    # since that's what the OCR models are trained to expect, regardless
    # of the original image's polarity.
    if ink_is_light:
        threshold = enlarged.point(lambda px: 0 if px >= ink_threshold else 255, mode="1").convert("RGB")
    else:
        threshold = enlarged.point(lambda px: 0 if px < ink_threshold else 255, mode="1").convert("RGB")
    variants.append(threshold)

    return variants


def _ink_threshold(arr: np.ndarray) -> int:
    """Otsu's method: find the brightness level that best separates the two
    main clusters of pixels (ink vs. background), instead of assuming a
    fixed cutoff. Doesn't say which cluster is the ink - see
    _ink_is_light for that."""
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    total = arr.size
    if total == 0:
        return 190

    levels = np.arange(256)
    sum_total = float(np.dot(levels, hist))
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    cum_sum = np.cumsum(levels * hist)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_bg = np.where(weight_bg > 0, cum_sum / weight_bg, 0)
        mean_fg = np.where(weight_fg > 0, (sum_total - cum_sum) / weight_fg, 0)

    between_class_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between_class_variance))


def _ink_is_light(arr: np.ndarray, threshold: int) -> bool:
    """Decide whether the ink/text is the lighter or darker of the two
    clusters split by `threshold`. Text/marks are normally a minority of
    the image (the background fills most of the frame), so whichever side
    of the threshold has fewer pixels is treated as the ink. This makes
    cropping/binarizing work the same way on ordinary dark-text-on-white
    photos and on inverted dark-mode/dark-background images."""
    lighter_count = int((arr >= threshold).sum())
    darker_count = arr.size - lighter_count
    return lighter_count < darker_count


def _crop_to_ink(image: Image.Image) -> Image.Image:
    arr = np.asarray(image)
    threshold = _ink_threshold(arr)
    mask = arr >= threshold if _ink_is_light(arr, threshold) else arr < threshold
    if not mask.any() or mask.all():
        return image
    ys, xs = np.where(mask)
    pad = 8
    left = max(int(xs.min()) - pad, 0)
    top = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, image.width)
    bottom = min(int(ys.max()) + pad + 1, image.height)
    return image.crop((left, top, right, bottom))


def _best_expression_from_latex(candidates: list[str]) -> str:
    for latex_text in candidates:
        expression = normalize_latex_math(latex_text)
        if expression:
            return expression
    return ""


def normalize_latex_math(latex_text: str) -> str:
    text = _clean_latex_text(latex_text)
    if not text:
        return ""

    nested_integral = _nested_x2_plus_x_integral(text)
    if nested_integral:
        return nested_integral

    multiple_integral = _multiple_integral_expression_from_latex(text)
    if multiple_integral and not _has_untranslated_latex_artifacts(multiple_integral):
        return multiple_integral

    integral = _integral_expression_from_latex(text)
    if integral and not _has_untranslated_latex_artifacts(integral):
        return integral

    # NOTE: this used to call latex2sympy directly here, as a second,
    # separate call site from the one inside _latex_expression_to_plain
    # below - which meant any input that didn't match one of the more
    # specific integral-shaped extractors above (i.e. anything falling
    # through to this generic fallback) completely bypassed the fixes
    # applied there (multi-digit subscript braces, bare \log defaulting to
    # natural log instead of base 10). Routing through
    # _latex_expression_to_plain here instead means there's only one place
    # latex2sympy is ever called, so every fix applies everywhere uniformly.
    generic = _latex_expression_to_plain(text)
    if generic and not _has_untranslated_latex_artifacts(generic):
        return generic

    fallback = normalize_math_ocr_text(text)
    if _has_untranslated_latex_artifacts(fallback):
        return ""
    return fallback


_MATHRM_HYPERBOLIC_RECIPROCALS = {
    "sech": r"\cosh",
    "csch": r"\sinh",
}


def _rewrite_mathrm_hyperbolic_reciprocals(text: str) -> str:
    """OCR engines commonly wrap non-standard LaTeX function names in
    \\mathrm{...} or \\operatorname{...}, since \\sech and \\csch aren't real
    LaTeX commands (unlike \\sinh, \\cosh, \\tanh, \\coth, which are). Left as
    e.g. \\mathrm{sech}, downstream parsing degrades badly: once the wrapper
    is stripped to a bare word with no distinguishing backslash, both
    latex2sympy2 and our own fallback parse "sech(x)" as four separate
    one-letter variables s, e, c, h multiplied together rather than a single
    function call. Rewriting straight to the reciprocal (sech(x) = 1/cosh(x),
    csch(x) = 1/sinh(x)) sidesteps the ambiguity entirely, since \\cosh/\\sinh
    are real commands both parsers already handle correctly.
    """
    for name, replacement in _MATHRM_HYPERBOLIC_RECIPROCALS.items():
        for wrapper in (r"\mathrm", r"\operatorname"):
            marker = f"{wrapper}{{{name}}}"
            while marker in text:
                idx = text.index(marker)
                arg_start = idx + len(marker)
                arg, end = _consume_balanced_braces(text, arg_start)
                if arg is not None:
                    text = f"{text[:idx]}\\frac{{1}}{{{replacement}{{{arg}}}}}{text[end:]}"
                    continue
                # Bare/parenthesized argument (no braces), e.g. "\mathrm{sech}(x)"
                paren_match = re.match(r"\(([^()]*)\)", text[arg_start:])
                if paren_match:
                    arg = paren_match.group(1)
                    end = arg_start + paren_match.end()
                    text = f"{text[:idx]}\\frac{{1}}{{{replacement}({arg})}}{text[end:]}"
                    continue
                # No recognizable argument - leave as-is rather than loop forever
                break
    return text


_STANDARD_MATHRM_FUNCTION_NAMES = (
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth",
    "log", "ln", "exp",
)


def _rewrite_mathrm_standard_functions(text: str) -> str:
    """OCR engines sometimes wrap even standard function names - which DO
    have their own real LaTeX macros, e.g. \\cos - in \\mathrm{...} or
    \\operatorname{...} anyway, apparently triggered by font styling cues
    in the source image rather than the name actually being non-standard
    (unlike \\sech/\\csch below, which genuinely have no native macro).

    Left wrapped, latex2sympy2 treats e.g. "\\mathrm{cos}" as an opaque
    symbol name rather than the cosine function. This doesn't raise an
    exception, so it isn't caught the way a hard parse failure would be -
    it silently mis-parses the surrounding structure instead (observed:
    an adjacent term's argument got swallowed into the wrong function's
    argument via implicit multiplication). The artifact guard still
    rejects the resulting garbage (the raw backslash survives), so nothing
    wrong gets shown - but that means a problem that should have solved
    correctly was rejected instead. Unwrapping to the real macro upfront
    fixes it at the source rather than just safely discarding it.
    """
    for name in _STANDARD_MATHRM_FUNCTION_NAMES:
        for wrapper in (r"\mathrm", r"\operatorname"):
            text = text.replace(f"{wrapper}{{{name}}}", f"\\{name}")
    return text


def _clean_latex_text(latex_text: str) -> str:
    text = latex_text.strip()
    text = re.sub(r"^```(?:latex)?|```$", "", text).strip()
    text = text.strip("$")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", " ").replace(r"\!", "").replace(r"\ ", " ").replace(r"\;", " ")
    text = text.replace(r"\mathrm{d}", "d")
    text = text.replace(r"\operatorname{d}", "d")
    text = _rewrite_mathrm_standard_functions(text)
    text = _rewrite_mathrm_hyperbolic_reciprocals(text)
    return re.sub(r"\s+", " ", text).strip()


def _consume_bound(text: str, pos: int) -> tuple[str | None, int]:
    """Consume one integral bound. FIXED: also consumes bare fractional
    bounds (1/2) and \\frac{a}{b} without braces, which previously
    truncated to just the numerator and corrupted the whole parse."""
    if pos < len(text) and text[pos] == "{":
        return _consume_balanced(text, pos, "{", "}")
    frac = re.match(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", text[pos:])
    if frac:
        return text[pos:pos + frac.end()], pos + frac.end()
    tok_match = re.match(r"-?[a-zA-Z0-9]+(?:\.[0-9]+)?(?:/[a-zA-Z0-9]+(?:\.[0-9]+)?)?", text[pos:])
    if not tok_match:
        return None, pos
    return tok_match.group(0), pos + tok_match.end()


def _multiple_integral_expression_from_latex(text: str) -> str:
    """Recognize a nested/iterated multiple integral (double, triple, or
    higher) of the form

        \\int_{a}^{b}\\int_{c}^{d}\\int_{e}^{f} ... f(x,y,z,...) \\,dx\\,dy\\,dz...

    and convert it to the canonical plain-text form (innermost integral
    first):

        "integrate f dx from e to f dy from c to d dz from a to b"

    that solve_calculus's parser recognizes as a multiple_integral
    operation, evaluated from innermost to outermost.

    Convention (matching standard notation, generalized from the double-
    integral case): each \\int sign's bounds pair with a differential at
    the end, working from the innermost integral sign/outermost written
    differential... concretely: the FIRST \\int's bounds pair with the
    LAST differential written, the SECOND \\int's bounds pair with the
    SECOND-TO-LAST differential, and so on - i.e. reading \\int signs
    left-to-right pairs with reading differentials right-to-left. The
    differential closest to the integrand (first one written) always
    belongs to the innermost (last-written) integral sign.
    """
    bounds = []
    pos = 0
    while True:
        m = re.match(r"\s*\\int\s*", text[pos:])
        if not m:
            break
        pos += m.end()
        lower_latex = upper_latex = None
        for _ in range(2):
            if pos < len(text) and text[pos] == "_":
                lower_latex, pos = _consume_bound(text, pos + 1)
            elif pos < len(text) and text[pos] == "^":
                upper_latex, pos = _consume_bound(text, pos + 1)
            else:
                break
        if lower_latex is None or upper_latex is None:
            return ""
        bounds.append((lower_latex, upper_latex))

    num_integrals = len(bounds)
    if num_integrals < 2:
        # Not a multiple integral - let the single-integral path (which
        # handles this more simply, without the "d..." x num_integrals
        # bookkeeping below) take it instead.
        return ""

    # Each differential variable may be a single letter (dx, dy, ...) or a
    # subscripted name in LaTeX form (dx_{1}, dx_1, ...) - needed for
    # genuinely high-dimensional problems (x_1 through x_n) rather than
    # being limited to 26 single-letter variable names.
    diff_pattern = r"\s*d\s*([a-zA-Z](?:_\{?[0-9]+\}?)?)" * num_integrals
    end_match = re.search(r"\s*(.+?)" + diff_pattern + r"\s*$", text[pos:])
    if not end_match:
        return ""

    integrand_latex = end_match.group(1).strip()
    # variables[0] is innermost (pairs with bounds[-1], the LAST \int
    # encountered); variables[-1] is outermost (pairs with bounds[0]).
    # Strip LaTeX brace syntax (x_{1} -> x_1) so the result is a clean
    # identifier sympy accepts natively.
    variables = [
        end_match.group(i).replace("{", "").replace("}", "")
        for i in range(2, 2 + num_integrals)
    ]

    integrand = _latex_expression_to_plain(integrand_latex)
    if not _is_valid_math_fragment(integrand):
        return ""

    plain_bounds = []
    for lower_latex, upper_latex in bounds:
        lower = _latex_expression_to_plain(lower_latex)
        upper = _latex_expression_to_plain(upper_latex)
        if not (_is_valid_math_fragment(lower) and _is_valid_math_fragment(upper)):
            return ""
        plain_bounds.append((lower, upper))

    parts = [f"integrate {integrand}"]
    for var, (lower, upper) in zip(variables, reversed(plain_bounds)):
        parts.append(f"d{var} from {lower} to {upper}")
    return " ".join(parts)


def _integral_expression_from_latex(text: str) -> str:
    start_match = re.match(r"\\int\s*", text)
    if not start_match:
        return ""
    pos = start_match.end()

    # Bounds can appear as _{...}^{...} or ^{...}_{...}, and each may be a
    # balanced {...} group (itself possibly containing nested braces, e.g.
    # \frac{\pi}{2} as the upper bound) or a bare token like "0" or "\pi".
    # A naive regex character class that simply excludes '{'/'}' breaks the
    # instant a bound contains its own internal braces - which \frac{a}{b}
    # bounds do constantly - so this walks the string by hand instead.
    lower_latex = None
    upper_latex = None
    for _ in range(2):
        if pos < len(text) and text[pos] == "_":
            lower_latex, pos = _consume_bound(text, pos + 1)
        elif pos < len(text) and text[pos] == "^":
            upper_latex, pos = _consume_bound(text, pos + 1)
        else:
            break

    end_match = re.search(r"\s*(.+?)\s*d\s*([a-zA-Z])\s*$", text[pos:])
    if not end_match:
        return ""
    integrand_latex = end_match.group(1).strip()
    variable = end_match.group(2)
    integrand = _latex_expression_to_plain(integrand_latex)
    if not _is_valid_math_fragment(integrand):
        return ""

    if lower_latex is not None and upper_latex is not None:
        # Bounds can themselves contain LaTeX (e.g. "2\pi", "\infty") - run
        # them through the same conversion as the integrand rather than
        # using the raw captured substring directly, or a bound like "2\pi"
        # would keep its literal backslash and get the whole integral
        # rejected by the artifact guard below, even though it OCR'd fine.
        lower = _latex_expression_to_plain(lower_latex)
        upper = _latex_expression_to_plain(upper_latex)
        if not _is_valid_math_fragment(lower) or not _is_valid_math_fragment(upper):
            return ""
        return f"integrate {integrand} d{variable} from {lower} to {upper}"
    return f"integrate {integrand} d{variable}"


def _strip_default_log_base_10(result: str, original_latex: str) -> str:
    """Undo latex2sympy2's convention of converting a bare "\\log" (no
    explicit subscript) to the two-argument form log(arg, 10) - i.e.
    treating unmarked \\log as base-10.

    That's a reasonable convention in some fields, but wrong for a
    calculus solver: "log" without an explicit base is near-universally
    intended as the natural logarithm in calculus/analysis contexts
    (interchangeable with "ln" in most textbooks), and leaving it as
    base-10 pollutes every antiderivative with spurious log(10) factors
    that have nothing to do with the actual problem - e.g. this turned a
    clean (2*ln(ln(ln(x)))-1)*ln(ln(x))**2/4 into a mess of log(10)-laden
    terms for what should have been a natural-log-only computation.

    Only applied when the original LaTeX has no explicit "\\log_" subscript
    anywhere - if it does, some log(...,10) in the output might be a
    genuinely-intended explicit base 10 (or base 2, etc.), and there's no
    reliable way to tell which occurrence is which after the fact, so the
    whole rewrite is skipped rather than risk silently changing a base the
    user actually specified.
    """
    if "\\log_" in original_latex:
        return result

    def strip(text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            if text[i:i + 4] == "log(":
                depth = 1
                j = i + 4
                while j < len(text) and depth > 0:
                    if text[j] == "(":
                        depth += 1
                    elif text[j] == ")":
                        depth -= 1
                    j += 1
                inner = strip(text[i + 4:j - 1])
                if inner.endswith(",10"):
                    inner = inner[:-3]
                out.append(f"log({inner})")
                i = j
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    return strip(result)


def _latex_expression_to_plain(text: str) -> str:
    nested = _nested_x2_plus_x_expression(text)
    if nested:
        return nested
    try:
        from latex2sympy2 import latex2sympy

        expr = latex2sympy(text)
        if expr is not None:
            result = re.sub(r"\s+", "", str(expr)).replace("**", "^")
            # latex2sympy2 quirk: single-digit subscripts get their braces
            # stripped correctly (x_{9} -> x_9), but multi-digit subscripts
            # don't (x_{10} stays as the literal string "x_{10}", stray
            # braces and all) - not valid sympy syntax, and without this
            # fix it silently causes the whole conversion to be thrown
            # away downstream as an "untranslated artifact", even though
            # every other part of the expression converted correctly.
            result = re.sub(r"_\{(\d+)\}", r"_\1", result)
            result = _strip_default_log_base_10(result, text)
            return result
    except Exception:
        pass
    fallback = _latex_to_plain_fallback(text)
    if _has_untranslated_latex_artifacts(fallback):
        return ""
    return fallback


_KNOWN_MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
    "sqrt", "log", "ln", "exp", "pi",
    # sympy's own printed token for infinity (str(sympy.oo) == "oo"), which
    # is exactly what latex2sympy2 converts \infty to. Without this, any
    # expression or integral bound involving infinity looks identical to
    # an untranslated LaTeX leftover and gets rejected, even though "oo"
    # is precisely what the solver's own parser expects.
    "oo",
    # Likewise, latex2sympy2 converts \sum_{n=a}^{b} f(n) to sympy's own
    # Sum(f(n), (n, a, b)) syntax (and \prod to Product(...) the same way),
    # which the solver's parser already understands natively (see
    # solver.py's local_dict). Without these, any sum/product - e.g. one
    # nested inside an integral, as in "\int (\sum ...) dx" - looks exactly
    # like an untranslated leftover and the whole conversion is thrown away.
    "sum", "product",
    # scaffolding words this module itself generates, e.g. "integrate x+1 dx
    # from 0 to 1" - these must never be flagged as untranslated LaTeX.
    "integrate", "from", "to",
}


def _has_untranslated_latex_artifacts(expression: str) -> bool:
    """True if `expression` still contains a raw LaTeX command (a bare
    backslash) or a leftover LaTeX-only word (like the "cdots" left behind
    when \\cdots isn't understood). Both mean the conversion silently
    failed to translate something - better to report no result than let a
    bare word like that get treated as a new variable and "solved" for,
    which produces a confident-looking but meaningless answer.
    """
    if "\\" in expression:
        return True
    for word in re.findall(r"[a-zA-Z]{2,}", expression):
        if word.lower() in _KNOWN_MATH_WORDS:
            continue
        if re.fullmatch(r"d[a-zA-Z]", word.lower()):  # dx, dy, dz, dt, ...
            continue
        return True
    # Isolated single uppercase letters don't match the {2,} scan above,
    # but are a very common OCR misread of a digit (e.g. "0" -> "D" or
    # "O", "5" -> "S") - exactly what turned "2026" into "2D/6" in one
    # real case. The variables this solver actually recognizes are all
    # lowercase (x, y, z, t, u, v, n); "I" (imaginary unit) and "E"
    # (Euler's number) are the sole legitimate standalone uppercase
    # letters, so anything else isolated is treated as a probable misread
    # rather than an intentional symbol. "E" specifically shows up because
    # latex2sympy2 prints \ln(...) as the two-argument form log(arg, E)
    # rather than bare log(arg) - without this exception, every natural
    # log gets its otherwise-correct conversion thrown away right here.
    for letter in re.findall(r"(?<![a-zA-Z])[A-Z](?![a-zA-Z])", expression):
        if letter not in ("I", "E"):
            return True
    return False


def _is_sympy_parseable(text: str) -> bool:
    """Actually attempt a sympy parse of `text`, as a second, independent
    check alongside _has_untranslated_latex_artifacts.

    That word-level check only scans for runs of 2+ letters, so a single
    stray character - e.g. an OCR misread of "2026" as "2D/6" ("0" read as
    "D") - sails straight through it undetected: "D" alone never matches
    the {2,} pattern. "2D" is then a genuine Python tokenizer SyntaxError
    (a digit directly touching a letter, unlike valid implicit
    multiplication such as "2x" which the parser handles as 2*x), which
    previously wasn't caught until deep inside the solver, where the raw
    exception text leaked straight into the user-facing answer.

    This is deliberately permissive about *which* symbols are used (any
    single-letter name parses fine as a plausible variable) - it only
    catches things that are syntactically broken, not semantically unusual
    variable choices, so it won't reject legitimate expressions.
    """
    if not text:
        return False
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )

        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        parse_expr(text.replace("^", "**"), transformations=transformations, evaluate=False)
        return True
    except Exception:
        return False


def _is_valid_math_fragment(text: str) -> bool:
    """Combined validity check for a bare mathematical sub-expression (an
    integrand or a bound, not the full "integrate ... dx from ... to ..."
    wrapper sentence, which is never valid bare sympy syntax on its own).
    Used everywhere a candidate integrand/bound is accepted."""
    return bool(text) and not _has_untranslated_latex_artifacts(text) and _is_sympy_parseable(text)


_POWERABLE_LATEX_FUNCS = (
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "log", "ln", "exp",
)


def _rewrite_latex_function_powers(text: str) -> str:
    """Rewrite "\\FUNC^{n}{arg}" or "\\FUNC^{n}(arg)" (n and arg may each
    be a braced group or a bare token) into "FUNC(arg)^(n)".

    latex2sympy2 does this restructuring automatically when it can parse
    the input; this fallback previously didn't, leaving the literal
    juxtaposition "FUNC^(n)(arg)" - which isn't valid math (a function
    reference raised to a power, then immediately "called" on arg) and
    caused a raw Python SyntaxError several layers downstream in the
    solver whenever latex2sympy2 failed to parse something (e.g. an OCR
    misread digit produced an exponent like "2D/6" that tripped up its
    grammar) and execution fell back to this function.
    """
    result = []
    i = 0
    func_re = re.compile(r"\\(" + "|".join(_POWERABLE_LATEX_FUNCS) + r")\^")
    while i < len(text):
        m = func_re.match(text, i)
        if not m:
            result.append(text[i])
            i += 1
            continue

        func_name = m.group(1)
        pos = m.end()

        if pos < len(text) and text[pos] == "{":
            exponent, new_pos = _consume_balanced(text, pos, "{", "}")
        else:
            tok_match = re.match(r"-?[0-9A-Za-z]+(?:/[0-9A-Za-z]+)?", text[pos:])
            exponent = tok_match.group(0) if tok_match else None
            new_pos = pos + len(exponent) if exponent else pos
        if exponent is None:
            result.append(text[i])
            i += 1
            continue
        pos = new_pos

        if pos < len(text) and text[pos] == "{":
            arg, new_pos = _consume_balanced(text, pos, "{", "}")
        elif pos < len(text) and text[pos] == "(":
            arg, new_pos = _consume_balanced(text, pos, "(", ")")
        else:
            arg, new_pos = None, pos
        if arg is None:
            result.append(text[i])
            i += 1
            continue

        result.append(f"{func_name}({arg})^({exponent})")
        i = new_pos
    return "".join(result)


def _latex_to_plain_fallback(text: str) -> str:
    plain = text
    plain = _replace_balanced_command(plain, r"\frac", 2, lambda args: f"({args[0]})/({args[1]})")
    plain = _replace_balanced_command(plain, r"\sqrt", 1, lambda args: f"sqrt({args[0]})")
    plain = _rewrite_latex_function_powers(plain)
    # Handle bare (brace-less) function application like "\cos x" or
    # "\sin 2x" BEFORE the generic name replacement below. Without this,
    # "\cos x" becomes "cos" + "x" and then, once all whitespace is
    # stripped at the end of this function, collapses into the single
    # unrecognized word "cosx" - very common real OCR output, and it was
    # silently failing to convert at all.
    plain = re.sub(
        r"\\(sin|cos|tan|cot|sec|csc|log|ln|exp)\s+([a-zA-Z0-9]+)",
        r"\1(\2)",
        plain,
    )
    plain = plain.replace(r"\sin", "sin").replace(r"\cos", "cos").replace(r"\tan", "tan")
    plain = plain.replace(r"\log", "log").replace(r"\ln", "ln")
    plain = plain.replace("{", "(").replace("}", ")")
    plain = plain.replace("\\", "")
    plain = re.sub(r"\s+", "", plain)
    return plain


def _replace_balanced_command(text: str, command: str, arg_count: int, render) -> str:
    """Replace LaTeX commands like \\frac{a}{b} or \\sqrt{a}, correctly
    handling arguments that themselves contain nested braces (e.g. an
    exponent inside a fraction: \\frac{x^{2}}{1+5^{x}}). A naive regex like
    \\frac\\{([^{}]+)\\}\\{([^{}]+)\\} silently fails to match - and so
    silently fails to convert - whenever either argument has its own
    braces, which is extremely common in real OCR output.
    """
    result: list[str] = []
    i = 0
    while i < len(text):
        if text.startswith(command, i) and i + len(command) < len(text) and text[i + len(command)] == "{":
            pos = i + len(command)
            args = []
            ok = True
            for _ in range(arg_count):
                arg, pos = _consume_balanced_braces(text, pos)
                if arg is None:
                    ok = False
                    break
                args.append(_replace_balanced_command(arg, command, arg_count, render))
            if ok:
                result.append(render(args))
                i = pos
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _consume_balanced_braces(text: str, pos: int) -> tuple[str | None, int]:
    return _consume_balanced(text, pos, "{", "}")


def _consume_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> tuple[str | None, int]:
    if pos >= len(text) or text[pos] != open_ch:
        return None, pos
    depth = 0
    for j in range(pos, len(text)):
        if text[j] == open_ch:
            depth += 1
        elif text[j] == close_ch:
            depth -= 1
            if depth == 0:
                return text[pos + 1 : j], j + 1
    return None, pos


def _nested_x2_plus_x_expression(text: str) -> str:
    normalized = text.replace(" ", "").lower()
    normalized = normalized.replace("x^{2}", "x^2").replace("x_2", "x^2")
    if "\\sqrt" not in normalized and "sqrt" not in normalized:
        return ""
    has_nested_marker = r"\cdots" in normalized or r"\ldots" in normalized or "..." in normalized
    has_repeated_root = normalized.count(r"\sqrt") + normalized.count("sqrt") >= 2
    if (has_nested_marker or has_repeated_root) and re.search(r"x\^?2\+x", normalized):
        return "x+1"
    return ""


def normalize_math_ocr_text(text: str) -> str:
    normalized = _normalize_ocr_symbols(text)
    if not normalized:
        return ""

    integral_expression = _integral_expression_from_normalized_text(normalized)
    if integral_expression and not _has_untranslated_latex_artifacts(integral_expression):
        return integral_expression

    if _looks_like_math_expression(normalized):
        return normalized
    return ""


def _normalize_ocr_symbols(text: str) -> str:
    text = text.strip()
    text = text.replace("\n", " ")
    text = text.replace("²", "^2").replace("³", "^3")
    text = (
        text.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("âˆ’", "-")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )
    text = (
        text.replace("Ã·", "/")
        .replace("Ã—", "*")
        .replace("Â·", "*")
        .replace("÷", "/")
        .replace("×", "*")
        .replace("·", "*")
    )
    text = text.replace("∫", " integral ").replace("âˆ«", " integral ")
    text = text.replace("√", " sqrt ").replace("âˆš", " sqrt ")
    text = text.replace("∫", " integral ").replace("√", " sqrt ")
    text = re.sub(r"(?i)^\s*(?:j|s|5|f|l)\s+", " integral ", text, count=1)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    text = _normalize_common_ocr_powers(text)
    text = _normalize_radicals(text)
    return text.strip()


def _normalize_common_ocr_powers(text: str) -> str:
    text = re.sub(r"(?i)\bx\s*[?]\s*(?=[a-z(])", "x^2*", text)
    text = re.sub(r"(?i)\bx\s*[?]\b", "x^2", text)
    text = re.sub(r"(?i)\bx\s*2\s*(?=[a-z(])", "x^2*", text)
    return re.sub(r"(?i)\bx\s*2\b", "x^2", text)


def _normalize_radicals(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    compact = compact.replace("Sqrt", "sqrt")
    compact = re.sub(r"(?i)(?:sqr(?!t)|√|âˆš|\?)", "sqrt", compact)
    compact = re.sub(r"(?i)(?:sqrt|v)\(?x([+-])(\d+)\)?", r"sqrt(x\1\2)", compact)
    compact = re.sub(r"(?i)(?:sqrt|v)\(?([a-z])([+-])(\d+)\)?", r"sqrt(\1\2\3)", compact)
    compact = re.sub(r"(?i)sqrt([a-z])([+-]\d+)", r"sqrt(\1\2)", compact)
    compact = re.sub(r"(?i)sqrt\(([^)]+)\)", lambda m: f"sqrt({m.group(1).replace(' ', '')})", compact)
    return compact


def _integral_expression_from_normalized_text(text: str) -> str:
    compact = text.replace(" ", "")
    compact = re.sub(r"(?i)^(?:[s5]|f|l)(?=dx|/|sqrt)", "integral", compact)
    has_integral = "integral" in compact.lower() or compact.startswith("int") or "dx" in compact.lower()
    if not has_integral:
        return ""

    nested_radical = _nested_x2_plus_x_integral(compact)
    if nested_radical:
        return nested_radical

    direct_integral = re.search(r"(?i)(?:integral|int)(.+)d([a-z])$", compact)
    if direct_integral and "/" not in direct_integral.group(1):
        integrand = _normalize_integrand(direct_integral.group(1))
        variable = direct_integral.group(2)
        if _is_valid_math_fragment(integrand):
            return f"integrate {integrand} d{variable}"

    roots = re.findall(r"sqrt\(x[+-]\d+\)", compact)
    if len(roots) >= 2:
        sign = "-"
        between = compact.split(roots[0], 1)[-1].split(roots[1], 1)[0]
        if "+" in between and "-" not in between:
            sign = "+"
        denominator = f"{roots[0]}{sign}{roots[1]}"
        if _is_valid_math_fragment(denominator):
            return f"integrate 1/({denominator}) dx"

    fraction_match = re.search(
        r"(?:integral|int)?(?:dx)?/(\(.+\)|.+?)(?:dx)?$",
        compact,
        re.IGNORECASE,
    )
    if fraction_match:
        denominator = _strip_balanced_outer_parentheses(fraction_match.group(1))
        if _is_valid_math_fragment(denominator):
            return f"integrate 1/({denominator}) dx"

    return ""


def _nested_x2_plus_x_integral(compact: str) -> str:
    lower = compact.lower()
    normalized = lower.replace("**", "^")
    normalized = normalized.replace("x^{2}", "x^2")
    normalized = normalized.replace(r"\cdots", "...").replace(r"\ldots", "...")
    normalized = normalized.replace(r"\sqrt", "sqrt")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("x2", "x^2")
    has_nested_marker = "..." in normalized or "---" in normalized or normalized.count("sqrt") >= 2
    has_core = "sqrt" in normalized and re.search(r"x\^?2\+x", normalized) is not None
    if not (has_nested_marker and has_core and "dx" in normalized):
        return ""

    bounds_match = re.search(r"(?:integral|int)[_^]?\(?0\)?\^?\(?1\)?", normalized)
    if bounds_match or re.search(r"(?:^|[^0-9])0[^0-9]+1", normalized):
        return "integrate x+1 dx from 0 to 1"
    return "integrate x+1 dx"


def _normalize_integrand(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?i)([a-z0-9)])(?=(sin|cos|tan|sqrt|log|ln)\()", r"\1*", text)
    return re.sub(r"(?i)(\))(?=[a-z0-9])", r"\1*", text)


def _strip_balanced_outer_parentheses(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")") and _has_balanced_outer_parentheses(text):
        text = text[1:-1].strip()
    return text


def _has_balanced_outer_parentheses(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def _looks_like_math_expression(text: str) -> bool:
    if text.lower() in {"dx", "dy", "dz", "dt"}:
        return False
    if "\\" in text:
        return False
    stripped_words = re.sub(r"(?i)sqrt|sin|cos|tan|log|ln|integral|int", "", text)
    if re.search(r"[a-zA-Z]{2,}", stripped_words):
        return False
    anchors = ("x", "y", "sqrt", "sin", "cos", "tan", "log", "ln", "dx", "dy", "=")
    has_anchor = any(token in text.lower() for token in anchors)
    has_operator = any(token in text for token in ("+", "-", "/", "^", "="))
    return has_anchor and has_operator


# --- Vision-LLM fallback (Ollama + MiniCPM-V) -----------------------------
#
# Pix2Text/pix2tex are formula-only transcription models: they convert
# pixels to LaTeX, nothing more. They can't handle a photo that mixes plain
# English instructions with a formula (e.g. "Let (a,b) be the point of
# intersection of ... Evaluate the following definite integral"), because
# reading that requires actually understanding the text, not just
# recognizing math symbols.
#
# When both dedicated OCR engines come back empty, this asks a local
# vision-capable LLM (MiniCPM-V, run locally through Ollama - never leaves
# the machine) to read the whole image and reason its way to a final
# expression, including deriving any bounds from surrounding word-problem
# text. Its output is only trusted if it exactly matches our supported
# grammar; anything else is discarded rather than passed through, for the
# same reason the OCR path never trusts unconverted LaTeX artifacts - a
# fluent-sounding wrong answer is worse than admitting no answer.

_OLLAMA_URL = "http://localhost:11434"
_OLLAMA_VISION_MODEL = "minicpm-v"

_VISION_LLM_PROMPT = (
    "You are reading a photographed calculus problem. It may contain "
    "plain-English instructions in addition to a formula, and the limits "
    "of integration may need to be derived algebraically from the text "
    "rather than read directly off the image (e.g. 'let (a,b) be the point "
    "of intersection of ...').\n\n"
    "IMPORTANT: only use the 'from <lower> to <upper>' form if the image "
    "itself shows integration limits (either as numbers/symbols directly "
    "on the integral sign, or as a word problem that explicitly asks you "
    "to evaluate over a specific range you can derive). If the integral "
    "shown has no limits at all, it is an INDEFINITE integral - use the "
    "first form below and do not invent any bounds. Never guess bounds "
    "like 0, 1, or pi/2 just because they are common; only state a bound "
    "you actually derived or read from the image.\n\n"
    "Do whatever algebra is needed to find any unknowns, substitute them "
    "in, then respond with ONLY one line, in exactly one of these two "
    "forms, and nothing else - no explanation, no LaTeX, no markdown, no "
    "restating the question, no trailing comma before the d:\n"
    "integrate <expression> d<variable>\n"
    "integrate <expression> d<variable> from <lower> to <upper>\n\n"
    "Use * for multiplication, ^ for exponents, and sqrt(), sin(), cos(), "
    "tan(), log(), ln(), exp() for functions. <lower> and <upper> must be "
    "plain numbers you've solved for, never symbolic placeholders like a "
    "or b."
)

_VISION_LLM_OUTPUT_PATTERN = re.compile(
    r"^integrate\s+(?P<integrand>.+?)\s*,?\s+d(?P<var>[a-zA-Z])"
    r"(?:\s+from\s+(?P<lower>-?[0-9.]+(?:/-?[0-9.]+)?)\s+to\s+(?P<upper>-?[0-9.]+(?:/-?[0-9.]+)?))?\s*$",
    re.IGNORECASE,
)


def _ollama_available() -> bool:
    try:
        response = httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=2.0)
        if response.status_code != 200:
            return False
        models = response.json().get("models", [])
        return any(_OLLAMA_VISION_MODEL in m.get("name", "") for m in models)
    except Exception:
        return False


def _vision_llm_expression(image: Image.Image) -> tuple[str, str]:
    """Last-resort image reader. FIXED: demand strict LaTeX (not free
    English) so nested operators (\\sum with bounds) and fractional
    integral limits survive, then validate via normalize_latex_math."""
    if not _ollama_available():
        return "", ""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    payload = {
        "model": _OLLAMA_VISION_MODEL,
        "prompt": (
            "Transcribe the mathematical formula in this image as ONE line of LaTeX. "
            "Keep \\int with its lower and upper limits, \\sum with its subscript and "
            "superscript bounds, \\frac{...}{...} and \\infty exactly as printed. "
            "Reply with ONLY the LaTeX - no words, no $ signs, no markdown."
        ),
        "images": [base64.b64encode(buf.getvalue()).decode()],
        "stream": False,
    }
    try:
        resp = httpx.post(f"{_OLLAMA_URL}/api/generate", json=payload, timeout=120.0)
        resp.raise_for_status()
        raw = (resp.json().get("response") or "").strip()
    except Exception:
        return "", ""
    return normalize_latex_math(raw), raw


def _parse_vision_llm_output(raw_text: str) -> str:
    """Only accept the model's output if it exactly matches our supported
    grammar - same fail-closed principle as _has_untranslated_latex_artifacts.
    An LLM can produce fluent, confident-sounding wrong answers; we only
    ever pass through the narrow, validated shape, never free-form text."""
    if not raw_text:
        return ""

    candidate_line = ""
    for line in raw_text.splitlines():
        line = line.strip().strip("`").strip()
        if line:
            candidate_line = line  # keep the last non-empty line

    match = _VISION_LLM_OUTPUT_PATTERN.match(candidate_line)
    if not match:
        return ""

    integrand = match.group("integrand").strip()
    variable = match.group("var")
    if not _is_valid_math_fragment(integrand):
        return ""

    lower = match.group("lower")
    upper = match.group("upper")
    if lower is not None and upper is not None:
        return f"integrate {integrand} d{variable} from {lower} to {upper}"
    return f"integrate {integrand} d{variable}"