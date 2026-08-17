"""Numeric verification of computed results.

This is the "never fabricate a result" safety net (item 14): every definite
integral, indefinite integral, and derivative gets checked numerically
against the original expression before being reported as verified, and any
genuine disagreement is reported honestly rather than papered over.

This replaced an earlier version that returned a fixed, topic-keyed string
(e.g. "Integral verified by differentiating the result...") no matter what
was actually computed - including for wrong answers. That's exactly what
happened with the cos(x)**2020 bug that motivated this module: the huge,
incorrect result was shown with a "✓ Verified" badge that had never actually
checked anything.
"""

import sympy as sp


def numeric_quad(expr, var, lower_val, upper_val):
    """Numerically integrate expr over [lower_val, upper_val], supplying
    breakpoints at integers (for floor()), at Abs()/Max()/Min() kink points,
    at Piecewise condition boundaries, and at period boundaries for
    periodic integrands - all of which introduce kinks, jump
    discontinuities, or high oscillation that plain adaptive quadrature can
    badly misjudge without explicit hints.

    Confirmed necessary by two real bugs found during development: frac(x)'s
    jump discontinuities at integers, and sin(x)**2 over a 200*pi interval
    (highly oscillatory relative to interval length) both gave silently
    wrong verification results without these breakpoints, even though the
    symbolic answers being checked were already correct.
    """
    import mpmath as mp

    # An unevaluated Sum/Product (e.g. from an integrand like
    # "Sum(x**n, (n, 2, oo))") lambdifies into raw Python sum()/range(),
    # which can't handle an infinite bound at all ("range(2, oo+1)" is
    # nonsensical) and throws deep inside mpmath's quadrature loop. Forcing
    # evaluation first collapses it to a closed form (here, a Piecewise
    # with x**2/(1-x) on the convergent region) that lambdifies normally.
    # A no-op for expressions with nothing to evaluate.
    try:
        expr = expr.doit()
    except Exception:
        pass

    f = sp.lambdify(var, expr, modules=["mpmath"])
    lo = mp.mpf(sp.N(lower_val))
    hi = mp.mpf(sp.N(upper_val))
    if lo > hi:
        lo, hi = hi, lo
    lo_f, hi_f = float(lo), float(hi)

    points = [lo, hi]

    if expr.has(sp.floor):
        import math
        start = math.ceil(lo_f)
        end = math.floor(hi_f)
        points.extend(float(n) for n in range(start, end + 1))

    try:
        from sympy.calculus.util import periodicity as _periodicity
        period = _periodicity(expr, var)
    except Exception:
        period = None
    if period and period > 0:
        period_f = float(period)
        span = hi_f - lo_f
        if span / period_f > 3:
            n_periods = int(span / period_f) + 1
            points.extend(
                p for k in range(n_periods + 1)
                if lo_f <= (p := lo_f + k * period_f) <= hi_f
            )

    def _add_solutions(equation):
        try:
            for sol in sp.solve(equation, var):
                if sol.is_real is False:
                    continue
                sval = float(sp.N(sol))
                if lo_f <= sval <= hi_f:
                    points.append(sval)
        except Exception:
            pass

    for a in expr.atoms(sp.Abs):
        _add_solutions(sp.Eq(a.args[0], 0))

    for m in list(expr.atoms(sp.Max)) + list(expr.atoms(sp.Min)):
        args = m.args
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                _add_solutions(sp.Eq(args[i], args[j]))

    if expr.has(sp.Piecewise):
        for pw in expr.atoms(sp.Piecewise):
            for _piece_expr, cond in pw.args:
                if cond is True:
                    continue
                for rel in cond.atoms(sp.core.relational.Relational):
                    _add_solutions(sp.Eq(rel.lhs, rel.rhs))

    points = sorted(set(points))
    return mp.quad(f, points)


def verify_ode_numeric(verify_ctx) -> tuple[bool, str]:
    """Numeric residual check for an ODE solution. Substitutes the candidate
    solution y(x) into the ODE's standard form (LHS = 0) and samples the
    residual at several points. A true solution leaves residual ~0 wherever
    the functions are defined.

    Returns a structured (ok: bool, message: str) pair so callers can make
    decisions on the verdict instead of sniffing message text. This is an
    independent numeric complement to the symbolic checkodesol test the
    ode_engine already runs.

    An arbitrary general solution still contains free constants C1, C2, ...
    The ODE must hold for ANY choice of them, so numeric values are pinned
    for the residual check (distinct, non-round numbers avoid masking bugs).
    """
    equation = verify_ctx["equation"]
    sol = verify_ctx["solution_expr"]  # Eq(y(x), rhs)
    var = verify_ctx["var"]            # x

    try:
        residual = (equation.lhs - equation.rhs).subs(sol.lhs, sol.rhs).doit()
        try:
            residual = sp.simplify(residual)
        except Exception:
            pass

        free = sorted(residual.free_symbols - {var}, key=lambda sym: sym.name)
        residual = residual.subs({
            sym: 1.3 + 0.1 * i for i, sym in enumerate(free)
        })

        f_res = sp.lambdify(var, residual, modules=["mpmath"])
    except Exception:
        return False, "Result computed symbolically; numeric ODE verification was not possible."

    sample_points = [-1.5, -0.6, 0.3, 0.7, 1.3, 2.1]
    checked, mismatches = 0, 0
    for p in sample_points:
        try:
            val = complex(f_res(p))
        except Exception:
            continue
        checked += 1
        if abs(val) > 1e-4:
            mismatches += 1

    if checked == 0:
        return False, "Result computed symbolically; numeric ODE verification unavailable for this domain."
    if mismatches == 0:
        return True, f"Verified numerically: the ODE residual is ~0 at {checked} sample points."
    return False, (
        f"⚠ Could not verify this result: substituting the solution into the ODE left a nonzero "
        f"residual at {mismatches}/{checked} sample points. Treat this answer with caution."
    )


def numerical_verify(verify_ctx, topic: str) -> str:
    """Numerically check the computed result against the original
    expression, and report honestly - including when the check fails or
    disagrees, rather than always claiming success.
    """
    if verify_ctx is None:
        return "Result computed via symbolic simplification (not independently numerically verified)."

    kind = verify_ctx.get("kind")

    try:
        if kind == "definite_integral":
            expr, var = verify_ctx["expr"], verify_ctx["var"]
            lower_val, upper_val, result = verify_ctx["lower"], verify_ctx["upper"], verify_ctx["result"]

            numeric_symbolic = complex(sp.N(result))
            numeric_quadrature = complex(numeric_quad(expr, var, lower_val, upper_val))

            if abs(numeric_symbolic.imag) < 1e-9 and abs(numeric_quadrature.imag) < 1e-9:
                a, b = numeric_symbolic.real, numeric_quadrature.real
                scale = max(abs(a), abs(b), 1.0)
                if abs(a - b) / scale < 1e-4:
                    return f"Verified numerically: symbolic result ≈ {a:.6g}, matches direct quadrature ({b:.6g})."
                return (
                    f"⚠ Could not verify this result: the symbolic answer evaluates to "
                    f"≈ {a:.6g}, but direct numerical integration gives ≈ {b:.6g}. "
                    f"Treat this answer with caution."
                )
            return "Result computed symbolically; numeric verification skipped (complex-valued)."

        if kind == "indefinite_integral":
            expr, var, antideriv = verify_ctx["expr"], verify_ctx["var"], verify_ctx["antideriv"]
            try:
                expr = expr.doit()
            except Exception:
                pass
            deriv_of_result = sp.diff(antideriv, var)
            sample_points = [0.3, 0.7, 1.3, -0.6, 2.1]
            f_expr = sp.lambdify(var, expr, modules=["mpmath"])
            f_check = sp.lambdify(var, deriv_of_result, modules=["mpmath"])

            checked, mismatches = 0, 0
            for p in sample_points:
                try:
                    a = complex(f_expr(p))
                    b = complex(f_check(p))
                except Exception:
                    continue
                checked += 1
                scale = max(abs(a), abs(b), 1.0)
                if abs(a - b) / scale > 1e-4:
                    mismatches += 1

            if checked == 0:
                return "Result computed symbolically; numeric verification unavailable for this expression's domain."
            if mismatches == 0:
                return f"Verified: differentiating the result reproduces the original integrand at {checked} sample points."
            return (
                f"⚠ Could not verify this result: differentiating the antiderivative disagreed with the "
                f"original integrand at {mismatches}/{checked} sample points. Treat this answer with caution."
            )

        if kind == "derivative":
            expr, var, result = verify_ctx["expr"], verify_ctx["var"], verify_ctx["result"]
            try:
                expr = expr.doit()
            except Exception:
                pass
            f_expr = sp.lambdify(var, expr, modules=["mpmath"])
            f_result = sp.lambdify(var, result, modules=["mpmath"])
            sample_points = [0.3, 0.7, 1.3, -0.6, 2.1]
            h = 1e-6

            checked, mismatches = 0, 0
            for p in sample_points:
                try:
                    finite_diff = (complex(f_expr(p + h)) - complex(f_expr(p - h))) / (2 * h)
                    symbolic_val = complex(f_result(p))
                except Exception:
                    continue
                checked += 1
                scale = max(abs(finite_diff), abs(symbolic_val), 1.0)
                if abs(finite_diff - symbolic_val) / scale > 1e-2:
                    mismatches += 1

            if checked == 0:
                return "Result computed symbolically; numeric verification unavailable for this expression's domain."
            if mismatches == 0:
                return f"Verified: matches a numeric finite-difference derivative at {checked} sample points."
            return (
                f"⚠ Could not verify this result: it disagreed with a numeric finite-difference check at "
                f"{mismatches}/{checked} sample points. Treat this answer with caution."
            )

        if kind == "ode":
            ok, message = verify_ode_numeric(verify_ctx)
            return message

    except Exception:
        # Verification machinery itself failed (e.g. couldn't lambdify some
        # exotic function) - be honest that we don't know, rather than
        # claiming either success or failure.
        return "Result computed symbolically; automatic numeric verification was not possible for this expression."

    if "Limit" in topic:
        return "Limit evaluated via symbolic methods (not independently numerically verified)."
    return "Result computed via symbolic manipulation and algebraic simplification (not independently numerically verified)."