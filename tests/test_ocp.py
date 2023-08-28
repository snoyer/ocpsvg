from typing import Iterable
import pytest
from pytest import raises
from OCP.Geom import Geom_BezierCurve, Geom_Curve
from OCP.gp import gp_Pnt

from ocpsvg.ocp import (
    bezier_curve,
    circle_curve,
    curve_to_beziers,
    curve_to_bspline,
    curve_to_polyline,
    ellipse_curve,
    segment_curve,
)
from ocpsvg.ocp import PntLike


@pytest.mark.parametrize(
    "a,b",
    [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 12)),
        ((0, 0, 0), (0, 12, 34)),
    ],
)
def test_segment_curve(a: PntLike, b: PntLike):
    assert isinstance(segment_curve(a, b), Geom_Curve)


@pytest.mark.parametrize(
    "a,b",
    [
        ((1, 2), (1, 2)),
        ((1, 2, 3), (1, 2, 3)),
    ],
)
def test_segment_curve_invalid(a: PntLike, b: PntLike):
    with raises(ValueError):
        assert isinstance(segment_curve(a, b), Geom_Curve)


@pytest.mark.parametrize(
    "controls",
    [
        [(0, 0), (10, 12)],
        [(0, 1), (1, 2), (3, 4)],
        [(1, 2, 3), (0, 12, 34)],
    ],
)
def test_bezier_curve(controls: Iterable[PntLike]):
    assert isinstance(bezier_curve(*controls), Geom_Curve)


@pytest.mark.parametrize(
    "controls",
    [
        [(1, 2)],
        [(i, i % 3) for i in range(26)],
    ],
)
def test_bezier_curve_invalid(controls: Iterable[PntLike]):
    with raises(ValueError):
        assert isinstance(bezier_curve(*controls), Geom_Curve)


VARIOUS_CURVES = [
    segment_curve((0, 0), (1, 1)),
    bezier_curve((0, 1), (23, 45), (67, 89)),
    bezier_curve((0, 1), (23, 45), (66, 89), (101, 213)),
    bezier_curve((0, 1), (23, 45), (66, 89), (101, 213), (141, 516)),
    curve_to_bspline(bezier_curve((0, 1), (23, 45), (66, 89), (101, 213), (141, 516))),
    circle_curve(12, clockwise=True),
    circle_curve(12, clockwise=False),
    circle_curve(12, 30, 210, clockwise=False),
    circle_curve(12, 30, 210, clockwise=True),
    ellipse_curve(12, 34, clockwise=True),
    ellipse_curve(12, 34, clockwise=False),
    ellipse_curve(12, 34, 30, 210, clockwise=False),
    ellipse_curve(12, 34, 30, 210, clockwise=True),
]


@pytest.mark.parametrize("curve", VARIOUS_CURVES)
def test_curve_to_beziers_deg3(curve: Geom_Curve):
    assert all(
        isinstance(e, Geom_BezierCurve) and e.Degree() <= 3
        for e in curve_to_beziers(curve, max_degree=3, tolerance=1e-6)
    )


@pytest.mark.parametrize("curve", VARIOUS_CURVES)
def test_curve_to_beziers_deg4(curve: Geom_Curve):
    assert all(
        isinstance(e, Geom_BezierCurve) and e.Degree() <= 4
        for e in curve_to_beziers(curve, max_degree=4, tolerance=1e-6)
    )


@pytest.mark.parametrize("curve", VARIOUS_CURVES)
def test_curve_to_polyline(curve: Geom_Curve):
    assert all(isinstance(p, gp_Pnt) for p in curve_to_polyline(curve, tolerance=1e-5))
