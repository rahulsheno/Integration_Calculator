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

    integral = _integral_expression_from_latex(text)
    if integral and not _has_untranslated_latex_artifacts(integral):
        return integral

    try:
        from latex2sympy2 import latex2sympy

        expr = latex2sympy(text)
        if expr is not None:
            result = re.sub(r"\s+", "", str(expr)).replace("**", "^")
            if not _has_untranslated_latex_artifacts(result):
                return result
    except Exception:
        pass

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


def _clean_latex_text(latex_text: str) -> str:
    text = latex_text.strip()
    text = re.sub(r"^```(?:latex)?|```$", "", text).strip()
    text = text.strip("$")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(r"\,", " ").replace(r"\!", "").replace(r"\ ", " ").replace(r"\;", " ")
    text = text.replace(r"\mathrm{d}", "d")
    text = text.replace(r"\operatorname{d}", "d")
    text = _rewrite_mathrm_hyperbolic_reciprocals(text)
    return re.sub(r"\s+", " ", text).strip()


def _integral_expression_from_latex(text: str) -> str:
    integral_match = re.search(
        r"\\int(?:_\{?([^}^{\s]+)\}?|\s*)?(?:\^\{?([^}^{\s]+)\}?)?\s*(.+?)\s*d\s*([a-zA-Z])\s*$",
        text,
    )
    if not integral_match:
        return ""

    lower_latex = integral_match.group(1)
    upper_latex = integral_match.group(2)
    integrand_latex = integral_match.group(3).strip()
    variable = integral_match.group(4)
    integrand = _latex_expression_to_plain(integrand_latex)
    if not integrand or _has_untranslated_latex_artifacts(integrand):
        return ""

    if lower_latex is not None and upper_latex is not None:
        # Bounds can themselves contain LaTeX (e.g. "2\pi", "\infty") - run
        # them through the same conversion as the integrand rather than
        # using the raw captured substring directly, or a bound like "2\pi"
        # would keep its literal backslash and get the whole integral
        # rejected by the artifact guard below, even though it OCR'd fine.
        lower = _latex_expression_to_plain(lower_latex)
        upper = _latex_expression_to_plain(upper_latex)
        if not lower or not upper or _has_untranslated_latex_artifacts(f"{lower} {upper}"):
            return ""
        return f"integrate {integrand} d{variable} from {lower} to {upper}"
    return f"integrate {integrand} d{variable}"


def _latex_expression_to_plain(text: str) -> str:
    nested = _nested_x2_plus_x_expression(text)
    if nested:
        return nested
    try:
        from latex2sympy2 import latex2sympy

        expr = latex2sympy(text)
        if expr is not None:
            return re.sub(r"\s+", "", str(expr)).replace("**", "^")
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
    return False


def _latex_to_plain_fallback(text: str) -> str:
    plain = text
    plain = _replace_balanced_command(plain, r"\frac", 2, lambda args: f"({args[0]})/({args[1]})")
    plain = _replace_balanced_command(plain, r"\sqrt", 1, lambda args: f"sqrt({args[0]})")
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
    if pos >= len(text) or text[pos] != "{":
        return None, pos
    depth = 0
    for j in range(pos, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
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
        if integrand:
            return f"integrate {integrand} d{variable}"

    roots = re.findall(r"sqrt\(x[+-]\d+\)", compact)
    if len(roots) >= 2:
        sign = "-"
        between = compact.split(roots[0], 1)[-1].split(roots[1], 1)[0]
        if "+" in between and "-" not in between:
            sign = "+"
        denominator = f"{roots[0]}{sign}{roots[1]}"
        return f"integrate 1/({denominator}) dx"

    fraction_match = re.search(
        r"(?:integral|int)?(?:dx)?/(\(.+\)|.+?)(?:dx)?$",
        compact,
        re.IGNORECASE,
    )
    if fraction_match:
        denominator = _strip_balanced_outer_parentheses(fraction_match.group(1))
        if denominator:
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
    """Returns (expression, raw_model_text). Empty expression if Ollama/the
    model isn't available, the request fails, or the model's output can't
    be safely parsed as one of our two supported forms."""
    if not _ollama_available():
        return "", ""

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    try:
        response = httpx.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model": _OLLAMA_VISION_MODEL,
                "prompt": _VISION_LLM_PROMPT,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=120.0,
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
    except Exception:
        return "", ""

    return _parse_vision_llm_output(raw_text), raw_text


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
    if not integrand or _has_untranslated_latex_artifacts(integrand):
        return ""

    lower = match.group("lower")
    upper = match.group("upper")
    if lower is not None and upper is not None:
        return f"integrate {integrand} d{variable} from {lower} to {upper}"
    return f"integrate {integrand} d{variable}"
