from typing import Iterable, Union

import pytest
from OCP.Geom import Geom_BezierCurve, Geom_Curve, Geom_TrimmedCurve
from OCP.gp import gp_Pnt, gp_Vec
from pytest import raises

from ocpsvg.ocp import (
    bezier_curve,
    circle_curve,
    curve_to_beziers,
    curve_to_bspline,
    curve_to_polyline,
    ellipse_curve,
    segment_curve,
)

XY = tuple[float, float]
XYZ = tuple[float, float, float]


def Pnt(x: float, y: float, z: float = 0):
    return gp_Pnt(x, y, z)


def as_Pnt(xy: XY):
    return gp_Pnt(*xy, 0)


def as_Pnts(*xys: XY):
    return map(as_Pnt, xys)


def as_tuple(v: Union[gp_Pnt, gp_Vec]):
    return v.X(), v.Y(), v.Z()


@pytest.mark.parametrize(
    "a,b",
    [
        (Pnt(0, 0), Pnt(1, 0)),
        (Pnt(0, 0), Pnt(0, 12)),
        (Pnt(0, 0, 0), Pnt(0, 12, 34)),
    ],
)
def test_segment_curve(a: gp_Pnt, b: gp_Pnt):
    assert isinstance(segment_curve(a, b), Geom_Curve)


@pytest.mark.parametrize(
    "a,b",
    [
        (Pnt(1, 2), Pnt(1, 2)),
        (Pnt(1, 2, 3), Pnt(1, 2, 3)),
    ],
)
def test_segment_curve_invalid(a: gp_Pnt, b: gp_Pnt):
    with raises(ValueError):
        assert isinstance(segment_curve(a, b), Geom_Curve)


@pytest.mark.parametrize(
    "controls",
    [
        [Pnt(0, 0), Pnt(10, 12)],
        [Pnt(0, 1), Pnt(1, 2), Pnt(3, 4)],
        [Pnt(1, 2, 3), Pnt(0, 12, 34)],
    ],
)
def test_bezier_curve(controls: Iterable[gp_Pnt]):
    assert isinstance(bezier_curve(*controls), Geom_Curve)


@pytest.mark.parametrize(
    "controls",
    [
        [Pnt(1, 2)],
        [Pnt(i, i % 3) for i in range(26)],
    ],
)
def test_bezier_curve_invalid(controls: Iterable[gp_Pnt]):
    with raises(ValueError):
        assert isinstance(bezier_curve(*controls), Geom_Curve)


VARIOUS_CURVES = [
    segment_curve(Pnt(0, 0), Pnt(1, 1)),
    bezier_curve(Pnt(0, 1), Pnt(23, 45), Pnt(67, 89)),
    bezier_curve(Pnt(0, 1), Pnt(23, 45), Pnt(66, 89), Pnt(101, 213)),
    bezier_curve(Pnt(0, 1), Pnt(23, 45), Pnt(66, 89), Pnt(101, 213), Pnt(141, 516)),
    curve_to_bspline(
        bezier_curve(Pnt(0, 1), Pnt(23, 45), Pnt(66, 89), Pnt(101, 213), Pnt(141, 516))
    ),
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
def test_curve_to_beziers(curve: Geom_Curve):
    tol = 1e-8
    bezs = list(curve_to_beziers(curve, tolerance=tol))
    assert bezs[0].StartPoint().IsEqual(curve.Value(curve.FirstParameter()), tol)
    assert bezs[-1].EndPoint().IsEqual(curve.Value(curve.LastParameter()), tol)


@pytest.mark.parametrize("curve", VARIOUS_CURVES)
def test_trimmed_curve_to_beziers(curve: Geom_Curve):
    u, v = curve.FirstParameter(), curve.LastParameter()
    trimmed = Geom_TrimmedCurve(curve, u + (v - u) * 0.25, u + (v - u) * 0.75)
    tol = 1e-8
    bezs = list(curve_to_beziers(trimmed, tolerance=tol))
    assert bezs[0].StartPoint().IsEqual(trimmed.Value(trimmed.FirstParameter()), tol)
    assert bezs[-1].EndPoint().IsEqual(trimmed.Value(trimmed.LastParameter()), tol)


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
