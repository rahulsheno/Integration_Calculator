"""Tests for the advanced-calculus engines: ODEs, Taylor/Maclaurin series,
series convergence tests, Laplace transforms, Fourier series, and vector
calculus (gradient/divergence/curl/Laplacian).

Run with: pytest tests/test_advanced.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import sympy as sp

from app.services.solver import solve_calculus
from app.services.ode_engine import solve_ode
from app.schemas.schemas import SolveRequest


def solve(expr: str):
    return solve_calculus(SolveRequest(expression=expr, include_graph=False))


class TestODEs:
    def test_second_order_homogeneous(self):
        r = solve("y'' + 4*y = 0")
        assert r.topic == "Differential Equations"
        # answer contains sin(2x) / cos(2x) family
        assert "sin(2*x)" in r.answer or "sin(2x)" in r.answer
        assert r.verification.startswith("Verified")

    def test_nonhomogeneous_with_forcing(self):
        r = solve("y'' + 4*y = sin(x)")
        assert "sin(x)/3" in r.answer
        assert r.verification.startswith("Verified")

    def test_separable(self):
        r = solve("y' = y^2 * x")
        assert r.topic == "Differential Equations"
        assert "1/(C1 + x^2)" in r.answer or "2/(C1 + x^2)" in r.answer
        assert r.verification.startswith("Verified")

    def test_ode_prefix(self):
        r = solve("ode y' + y = 0")
        assert r.topic == "Differential Equations"
        assert "exp(-x)" in r.answer

    def test_ivp(self):
        r = solve("ode y'' + y = 0 with y(0) = 1, y'(0) = 0")
        # y = cos(x) is the exact IVP solution
        ans = r.answer.replace(" ", "")
        assert "cos(x)" in ans
        assert "C1" not in ans

    def test_unsolvable_reported_honestly(self):
        # dy/dx = y^2 + x has no elementary closed form
        result = solve_ode("y' = y^2 + x")
        assert result is not None
        assert result["solution"] is None
        assert result["error"]

    def test_leibniz_notation_parsed(self):
        from app.services.solver import parse_expression
        assert parse_expression("dy/dx = x*y")[0] == "differential_equation"


class TestTaylorSeries:
    def test_maclaurin_exp(self):
        r = solve("taylor exp(x) order 4")
        assert r.topic == "Taylor & Maclaurin Series"
        assert "x^4/24" in r.answer

    def test_taylor_about_nonzero_center(self):
        r = solve("taylor log(x) at x=1 order 3")
        assert r.topic == "Taylor & Maclaurin Series"
        # series of log about 1: (x-1) - (x-1)^2/2 + (x-1)^3/3
        assert "x - 1" in r.answer or "(x - 1)" in r.answer


class TestConvergence:
    def test_harmonic_diverges(self):
        r = solve("convergence of sum 1/n from n=1 to oo")
        assert r.answer == "Diverges"
        assert "Integral Test" in r.verification

    def test_p_series_converges(self):
        r = solve("convergence of sum 1/n^2 from n=1 to oo")
        assert r.answer == "Converges"

    def test_ratio_test_decisive(self):
        r = solve("convergence of sum n/2^n from n=1 to oo")
        assert r.answer == "Converges"
        assert "Ratio Test" in r.verification

    def test_nth_term_divergence(self):
        r = solve("convergence of sum n/(n+1) from n=1 to oo")
        assert r.answer == "Diverges"
        assert "Nth-Term Test" in r.verification


class TestLaplace:
    def test_exponential(self):
        r = solve("laplace exp(3*x)")
        assert r.answer == "1/(s - 3)"

    def test_inverse(self):
        r = solve("inverse laplace 1/(s^2 + 1)")
        assert r.answer == "sin(x)"

    def test_inverse_shifted(self):
        r = solve("inverse laplace (s+1)/(s^2 + 2*s + 5)")
        assert "exp(-x)*cos(2*x)" in r.answer


class TestFourier:
    def test_odd_function_sine_series(self):
        r = solve("fourier x on [-pi, pi]")
        assert r.topic == "Fourier Series"
        assert "sin(x)" in r.answer
        assert "cos" not in r.answer


class TestVectorCalculus:
    def test_gradient(self):
        r = solve("gradient of x^2*y + z")
        assert r.topic == "Vector Calculus"
        assert "2*x*y" in r.answer
        assert r.verification.startswith("Verified")

    def test_divergence(self):
        r = solve("divergence of x*y, y*z, z*x")
        assert r.answer == "x + y + z"
        assert r.verification.startswith("Verified")

    def test_curl_conservative_field_is_zero(self):
        # F = grad(xyz) = (yz, xz, xy) => curl must be zero
        r = solve("curl of y*z, x*z, x*y")
        assert "0" in r.answer
        assert r.verification.startswith("Verified")

    def test_laplacian(self):
        r = solve("laplacian of x^2 + y^2 + 3*z^2")
        assert r.answer == "10"
        assert r.verification.startswith("Verified")

    def test_harmonic_function_has_zero_laplacian(self):
        r = solve("laplacian of exp(x)*sin(y)")
        assert sp.simplify(sp.sympify(r.answer)) == 0
