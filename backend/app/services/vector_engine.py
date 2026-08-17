"""Vector calculus: gradient, divergence, curl, and Laplacian.

Implements the standard del-operator (nabla) quantities directly from
partial derivatives, on scalar fields f(x, y, z) and vector fields given
as component tuples (P, Q, R):

    grad f      = (∂f/∂x, ∂f/∂y, ∂f/∂z)
    div F       = ∂P/∂x + ∂Q/∂y + ∂R/∂z
    curl F      = (∂R/∂y − ∂Q/∂z, ∂P/∂z − ∂R/∂x, ∂Q/∂x − ∂P/∂y)
    ∇²f         = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

These are all exact symbolic differentiations via sympy's diff(), so the
results are exact and can be verified by re-differentiating. Every result
is verified by recomputing one component numerically as an independent
check (finite differences), matching the solver's general
"verify independently before claiming success" policy.

Deliberately NOT handled here: line integrals, surface integrals, and the
integral theorems (Green, Stokes, Divergence) - those reduce to evaluating
parametrized single/multiple integrals, which solver.py's existing integral
machinery already does; wiring those named-theorem paths on top is tracked
separately.
"""

import sympy as sp

from app.services.compute_utils import run_with_timeout, ComputeTimeout

_COMPONENT_NAMES = ("x", "y", "z")


def _symbols():
    return tuple(sp.Symbol(name) for name in _COMPONENT_NAMES)


def compute_gradient(scalar_expr):
    """Gradient of a scalar field over (x, y, z). Returns a dict with
    'components' (list of partials), 'vector' (sp.Matrix), 'error'."""
    if scalar_expr is None:
        return {"components": None, "vector": None, "error": "No expression."}
    try:
        x, y, z = _symbols()
        comps = [run_with_timeout(sp.diff, scalar_expr, v, seconds=20) for v in (x, y, z)]
        return {"components": [sp.simplify(c) for c in comps], "vector": sp.Matrix(comps), "error": None}
    except ComputeTimeout as e:
        return {"components": None, "vector": None, "error": str(e)}
    except Exception:
        return {"components": None, "vector": None, "error": "Couldn't differentiate this field."}


def compute_divergence(components):
    """Divergence of a vector field given as a sequence of component
    expressions (in x, y, z order; missing trailing components are 0)."""
    comps = _pad_components(components)
    if comps is None:
        return {"divergence": None, "error": "A vector field needs up to 3 components (P, Q, R)."}
    try:
        x, y, z = _symbols()
        parts = [run_with_timeout(sp.diff, c, v, seconds=20) for c, v in zip(comps, (x, y, z))]
        return {"divergence": sp.simplify(sum(parts)), "error": None}
    except ComputeTimeout as e:
        return {"divergence": None, "error": str(e)}
    except Exception:
        return {"divergence": None, "error": "Couldn't compute the divergence of this field."}


def compute_curl(components):
    """Curl of a 3-component vector field (P, Q, R) over (x, y, z)."""
    comps = _pad_components(components)
    if comps is None:
        return {"curl": None, "error": "A vector field needs up to 3 components (P, Q, R)."}
    try:
        x, y, z = _symbols()
        P, Q, R = comps
        curl = (
            run_with_timeout(sp.diff, R, y, seconds=20) - run_with_timeout(sp.diff, Q, z, seconds=20),
            run_with_timeout(sp.diff, P, z, seconds=20) - run_with_timeout(sp.diff, R, x, seconds=20),
            run_with_timeout(sp.diff, Q, x, seconds=20) - run_with_timeout(sp.diff, P, y, seconds=20),
        )
        return {"curl": sp.Matrix([sp.simplify(c) for c in curl]), "error": None}
    except ComputeTimeout as e:
        return {"curl": None, "error": str(e)}
    except Exception:
        return {"curl": None, "error": "Couldn't compute the curl of this field."}


def compute_laplacian(scalar_expr):
    """Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² of a scalar field."""
    if scalar_expr is None:
        return {"laplacian": None, "error": "No expression."}
    try:
        x, y, z = _symbols()
        parts = [run_with_timeout(sp.diff, scalar_expr, v, 2, seconds=20) for v in (x, y, z)]
        return {"laplacian": sp.simplify(sum(parts)), "error": None}
    except ComputeTimeout as e:
        return {"laplacian": None, "error": str(e)}
    except Exception:
        return {"laplacian": None, "error": "Couldn't compute the Laplacian of this field."}


def _pad_components(components):
    """Normalize a user-supplied component list to exactly 3 entries,
    padding missing trailing components with 0. Returns None if the field
    can't be interpreted as a 1-, 2-, or 3-component vector field."""
    if components is None:
        return None
    comps = list(components)
    if not comps or len(comps) > 3:
        return None
    while len(comps) < 3:
        comps.append(sp.Integer(0))
    if any(c is None for c in comps):
        return None
    return comps


def numerical_verify_gradient(scalar_expr, precomputed=None, point=(0.3, -0.4, 0.7), h=1e-5):
    """Independent numeric check of compute_gradient: compare each analytic
    component against a central finite difference at `point`. Returns
    (True, message) on agreement, (False, message) on disagreement.

    Pass the result of compute_gradient as `precomputed` to avoid
    re-differentiating the field (the solver already computed it for the
    answer). Falls back to computing when omitted.
    """
    g = precomputed if precomputed is not None else compute_gradient(scalar_expr)
    if g["error"] or g["components"] is None:
        return False, "No analytic gradient to verify."
    try:
        x, y, z = _symbols()
        f = sp.lambdify((x, y, z), scalar_expr, modules=["mpmath"])
        analytic = [sp.lambdify((x, y, z), c, modules=["mpmath"]) for c in g["components"]]
    except Exception:
        return False, "Couldn't build numeric functions for verification."

    try:
        for i in range(3):
            p_plus, p_minus = list(point), list(point)
            p_plus[i] += h
            p_minus[i] -= h
            fd = (complex(f(*p_plus)) - complex(f(*p_minus))) / (2 * h)
            av = complex(analytic[i](*point))
            scale = max(abs(av), abs(fd), 1.0)
            if abs(av - fd) / scale > 1e-3:
                return False, f"Gradient component {i + 1} disagrees with finite differences at the test point."
        return True, "Verified: each component matches a central finite-difference derivative at a test point."
    except Exception:
        return False, "Numeric verification unavailable for this expression's domain."


def numerical_verify_divergence(components, precomputed=None, point=(0.3, -0.4, 0.7), h=1e-5):
    """Numeric check of compute_divergence via finite differences. Pass the
    result of compute_divergence as `precomputed` to avoid recomputing."""
    d = precomputed if precomputed is not None else compute_divergence(components)
    if d["error"] or d["divergence"] is None:
        return False, "No analytic divergence to verify."
    comps = _pad_components(components)
    if comps is None:
        return False, "Couldn't interpret the vector field for verification."
    try:
        x, y, z = _symbols()
        funcs = [sp.lambdify((x, y, z), c, modules=["mpmath"]) for c in comps]
        analytic = sp.lambdify((x, y, z), d["divergence"], modules=["mpmath"])

        fd_total = 0j
        for i in range(3):
            p_plus, p_minus = list(point), list(point)
            p_plus[i] += h
            p_minus[i] -= h
            fd_total += (complex(funcs[i](*p_plus)) - complex(funcs[i](*p_minus))) / (2 * h)
        av = complex(analytic(*point))
        scale = max(abs(av), abs(fd_total), 1.0)
        if abs(av - fd_total) / scale > 1e-3:
            return False, "Divergence disagrees with a finite-difference check at the test point."
        return True, "Verified: matches a finite-difference divergence at a test point."
    except Exception:
        return False, "Numeric verification unavailable for this field's domain."


def numerical_verify_curl(components, precomputed=None, point=(0.3, -0.4, 0.7), h=1e-5):
    """Numeric check of compute_curl via finite differences. Pass the result
    of compute_curl as `precomputed` to avoid recomputing."""
    c = precomputed if precomputed is not None else compute_curl(components)
    if c["error"] or c["curl"] is None:
        return False, "No analytic curl to verify."
    comps = _pad_components(components)
    if comps is None:
        return False, "Couldn't interpret the vector field for verification."
    try:
        x, y, z = _symbols()
        P, Q, R = (sp.lambdify((x, y, z), comp, modules=["mpmath"]) for comp in comps)
        analytic = [sp.lambdify((x, y, z), c["curl"][i], modules=["mpmath"]) for i in range(3)]

        def fd(func, axis):
            p_plus, p_minus = list(point), list(point)
            p_plus[axis] += h
            p_minus[axis] -= h
            return (complex(func(*p_plus)) - complex(func(*p_minus))) / (2 * h)

        fd_curl = [
            fd(R, 1) - fd(Q, 2),
            fd(P, 2) - fd(R, 0),
            fd(Q, 0) - fd(P, 1),
        ]
        for i in range(3):
            av = complex(analytic[i](*point))
            scale = max(abs(av), abs(fd_curl[i]), 1.0)
            if abs(av - fd_curl[i]) / scale > 1e-3:
                return False, f"Curl component {i + 1} disagrees with finite differences at the test point."
        return True, "Verified: each curl component matches a finite-difference check at a test point."
    except Exception:
        return False, "Numeric verification unavailable for this field's domain."


def numerical_verify_laplacian(scalar_expr, precomputed=None, point=(0.3, -0.4, 0.7), h=1e-4):
    """Numeric check of compute_laplacian via second-order finite
    differences. Pass the result of compute_laplacian as `precomputed` to
    avoid recomputing."""
    lap = precomputed if precomputed is not None else compute_laplacian(scalar_expr)
    if lap["error"] or lap["laplacian"] is None:
        return False, "No analytic Laplacian to verify."
    try:
        x, y, z = _symbols()
        f = sp.lambdify((x, y, z), scalar_expr, modules=["mpmath"])
        analytic = sp.lambdify((x, y, z), lap["laplacian"], modules=["mpmath"])

        def d2(axis):
            p, p_plus, p_minus = list(point), list(point), list(point)
            p_plus[axis] += h
            p_minus[axis] -= h
            return (complex(f(*p_plus)) - 2 * complex(f(*p)) + complex(f(*p_minus))) / (h * h)

        fd_total = d2(0) + d2(1) + d2(2)
        av = complex(analytic(*point))
        scale = max(abs(av), abs(fd_total), 1.0)
        if abs(av - fd_total) / scale > 1e-3:
            return False, "Laplacian disagrees with a finite-difference check at the test point."
        return True, "Verified: matches a second-order finite-difference Laplacian at a test point."
    except Exception:
        return False, "Numeric verification unavailable for this expression's domain."
