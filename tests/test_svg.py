import re
from io import StringIO
from math import pi
from typing import Any, Sequence, Union

import pytest
from OCP.Geom import Geom_BezierCurve, Geom_Curve
from OCP.gp import gp_Vec
from OCP.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Wire
from pytest import approx

from ocpsvg.ocp import (
    PntLike,
    as_Pnt,
    as_triple,
    bezier_curve,
    bounding_box,
    circle_curve,
    edge_from_curve,
    ellipse_curve,
    face_inner_wires,
    is_wire_closed,
    segment_curve,
)
from ocpsvg.svg import (
    ColorAndLabel,
    SvgPathCommand,
    bezier_to_svg_path,
    edge_to_svg_path,
    edges_from_svg_path,
    faces_from_svg_path,
    format_svg,
    import_svg_document,
    polyline_to_svg_path,
    wire_to_svg_path,
    wires_from_svg_path,
)

from .ocp import face_area, face_normal, wire_from_edges


class SvgPath(list[SvgPathCommand]):
    def __str__(self) -> str:
        return format_svg(self)


@pytest.mark.parametrize(
    "points, closed, svg_d",
    [
        ([(0, 1), (2, 3)], False, "M0,1 L 2,3"),
        ([(0, 1), (2, 3), (2, 5)], False, "M0,1 L 2,3 L 2,5"),
        ([(0, 1), (2, 3), (2, 5), (5, 5)], False, "M0,1 L 2,3 L 2,5 L 5,5"),
        ([(0, 1), (2, 3), (2, 5), (5, 5)], True, "M0,1 L 2,3 L 2,5 L 5,5 Z"),
    ],
)
def test_polyline_to_svg(points: Sequence[PntLike], closed: bool, svg_d: str):
    path = SvgPath(polyline_to_svg_path(map(as_Pnt, points), closed=closed))
    assert svg_path_tokens(path) == approx(svg_path_tokens(svg_d), abs=1e-4), str(path)


@pytest.mark.parametrize(
    "bez_curve, svg_d, opts",
    [
        (bezier_curve((0, 1), (2, 3)), "M0,1 L 2,3", {}),
        (bezier_curve((0, 1), (2, 3), (2, 5)), "M0,1 Q 2,3 2,5", {}),
        (bezier_curve((0, 1), (2, 3), (2, 5), (5, 5)), "M0,1 C 2,3 2,5 5,5", {}),
    ],
)
def test_bezier_to_svg(bez_curve: Geom_BezierCurve, svg_d: str, opts: dict[str, Any]):
    path = SvgPath(bezier_to_svg_path(bez_curve, **opts))
    assert svg_path_tokens(path) == approx(svg_path_tokens(svg_d), abs=1e-4), str(path)


def test_bezier_to_svg_error():
    bez_curve = bezier_curve((0, 1), (2, 3), (2, 5), (5, 5), (10, 5))
    with pytest.raises(ValueError):
        SvgPath(bezier_to_svg_path(bez_curve))


@pytest.mark.parametrize(
    "curve, svg_d, opts",
    [
        (segment_curve((0, 1), (2, 3)), "M0,1 L 2,3", {}),
        (bezier_curve((0, 1), (3, 2), (4, 5)), "M 0,1 Q 3,2,4,5", {}),
        (
            bezier_curve((0, 1), (3, 2), (4, 5)),
            "M 0.0,1.0 C 2.0,1.666666,3.333333,3.0,4.0,5.0",
            dict(use_quadratics=False),
        ),
        (bezier_curve((0, 1), (3, 2), (4, 5), (7, 8)), "M 0,1 C 3,2,4,5,7,8", {}),
        (circle_curve(2), "M2,0 A2,2,180,1,1,-2,0 A2,2,180,1,1,2,0", {}),
        (circle_curve(2), "M2,0 A2,2,359.9999,1,0,2,-0", dict(split_full_arcs=False)),
        (circle_curve(2, 90, 180), "M-2.0,0 A2.0,2.0,0.0,0,0,0,2.0", {}),
        (ellipse_curve(3, 1), "M3,0 A3,1,360,1,0,3,0", dict(split_full_arcs=False)),
        (ellipse_curve(3, 1), "M3,0 A3,1,180,1,1,-3,0 A3,1,180,1,1,3,0", {}),
        (ellipse_curve(3, 1, 90, 180), "M-3.0,0 A3.0,1.0,0.0,0,0,0,1.0", {}),
    ],
)
def test_edge_to_svg(curve: Geom_Curve, svg_d: str, opts: dict[str, Any]):
    edge = edge_from_curve(curve)
    path = SvgPath(edge_to_svg_path(edge, tolerance=1e-5, **opts))
    assert svg_path_tokens(path) == approx(svg_path_tokens(svg_d), abs=1e-4), str(path)


@pytest.mark.parametrize(
    "curve",
    [
        segment_curve((0, 1), (2, 3)),
        bezier_curve((0, 1), (3, 2), (4, 5)),
        bezier_curve((0, 1), (3, 2), (4, 5), (7, 8)),
        circle_curve(2),
        circle_curve(2, 90, 180),
        ellipse_curve(3, 1, 90, 180),
    ],
)
def test_polyline_apprx(curve: Geom_Curve):
    edge = edge_from_curve(curve)
    path = SvgPath(
        edge_to_svg_path(
            edge, tolerance=1e-5, use_arcs=False, use_cubics=False, use_quadratics=False
        )
    )
    assert set(seg[0] for seg in path).issubset(set("MLZ"))


@pytest.mark.parametrize(
    "curves, svg_d, opts",
    [
        (
            [segment_curve((-1, 4), (0, 1)), bezier_curve((0, 1), (3, 2), (4, 5))],
            "M-1,4 L 0,1" "Q 3,2,4,5",
            {},
        ),
        (
            [segment_curve((-1, 4), (0, 1)), bezier_curve((0, 1), (3, 2), (4, 5))],
            "M-1,4 L 0,1" "C 2.0,1.666666,3.333333,3.0,4.0,5.0",
            dict(use_quadratics=False),
        ),
    ],
)
def test_wire_to_svg(curves: list[Geom_Curve], svg_d: str, opts: dict[str, Any]):
    wire = wire_from_edges(map(edge_from_curve, curves))
    path = SvgPath(wire_to_svg_path(wire, tolerance=1e-5, **opts))
    assert svg_path_tokens(path) == approx(svg_path_tokens(svg_d), abs=1e-4), str(path)


def test_arc_flags():
    c = pi * 45**2
    s = (45 * 2) ** 2
    cases = [
        ("M  80  80 A 45 45, 0, 0, 0, 125 125 L 125  80 Z", 1 / 4 * c),
        ("M 230  80 A 45 45, 0, 1, 0, 275 125 L 275  80 Z", 3 / 4 * c + 1 / 4 * s),
        ("M  80 230 A 45 45, 0, 0, 1, 125 275 L 125 230 Z", 1 / 4 * (s - c)),
        ("M 230 230 A 45 45, 0, 1, 1, 275 275 L 275 230 Z", 3 / 4 * c),
    ]
    for path, area in cases:
        res = list(faces_from_svg_path(path))
        assert len(res) == 1
        assert isinstance(res[0], TopoDS_Face)
        assert face_area(res[0]) == approx(area)


def test_path_regression1():
    """this path used to generate segments below tolerance when being closed"""
    d = """
    M 33.5373470102,42.5176389595
    A 4.09636,4.09636 0.0 0,1 29.4409888981,46.6139970715
    A 4.09636,4.09636 0.0 0,1 25.3446307861,42.5176389595
    A 4.09636,4.09636 0.0 0,1 29.4409888981,38.4212808475
    A 4.09636,4.09636 0.0 0,1 33.5373470102,42.5176389595
    Z
    """
    wires = list(wires_from_svg_path(d))
    assert len(wires) == 1
    assert is_wire_closed(wires[0])


def test_path_regression2():
    """the small arc in this path used to no be handled right"""
    d = """
    m 11.76429,44.366052 
    c 0.865598,-0.538449 2.987334,-1.654154 5.219562,-3.0173931.860913,-1.136474
    4.482602,-2.785758 7.441652,-4.045502 2.033322,-0.865638 4.174966,-1.53082
    6.465079,-1.859277 2.328204,-0.333919 4.652868,-0.296857 7.012161,0.09842 
    0.0066,0.0016 0.01317,0.0032 0.01975,0.0049 1.236231,0.306435 2.433142,0.808671
    3.617195,1.509105 1.146272,0.678084 2.25166,1.524577 3.349647,2.518228 
    2.211881,2.001698 4.261035,4.482385 6.312848,6.896273 2.52664,2.972508 
    4.591476,5.255564 6.895181,7.028956 0.52806,0.406501 1.03584,0.75893 
    1.522741,1.061625 0.901859,0.560664 1.808731,0.981731 2.68984,1.286418 
    2.090466,0.722882 3.751425,0.698313 4.665988,0.720769 -0.895985,-0.382042 
    -2.163264,-1.190245 -3.678877,-2.48684 -0.641461,-0.548765 -1.26771,-1.128549 
    -1.959619,-1.780292 -0.371036,-0.349498 -0.763728,-0.727866 -1.188041,-1.152766 
    -2.037319,-2.040141 -3.506627,-3.812847 -6.177306,-7.011402 -1.988611,-2.381671 
    -4.248901,-5.050679 -6.738801,-7.211117 -1.257303,-1.090937 -2.575615,-2.057722 
    -3.984539,-2.846659 -1.462016,-0.818666 -2.971376,-1.417477 -4.550841,-1.782702 
    -0.02614,-0.006 -0.05233,-0.01203 -0.07858,-0.01796
    a 0.38591405,0.03021177 15.61296 0 0 -0.113793,-0.0224
    c -2.725017,-0.415945 -5.42773,-0.409008 -8.116554,0.06309 -2.609069,0.45809 
    -4.995593,1.317744 -7.161968,2.400596 -3.147128,1.573078 -5.751218,3.587805 
    -7.400338,5.091117 -1.968561,1.794511 -3.433935,3.806292 -4.062386,4.554845
    z
    """
    wires = list(wires_from_svg_path(d))
    assert len(wires) == 1
    assert is_wire_closed(wires[0])


def test_arcs_path_to_wire():
    """this continuous path introduces small discontinuities when making the edges"""
    res = list(
        wires_from_svg_path(
            "M 10 315 L 110 215"
            "A 30 50 0 0 1 162.55 162.45"
            "L 172.55 152.45"
            "A 30 50 -45 0 1 215.1 109.9"
            "L 315 10"
        )
    )
    assert len(res) == 1
    assert isinstance(res[0], TopoDS_Wire)


@pytest.mark.parametrize(
    "svg_d",
    [
        "M 10 5 L 15 5 L 15 5 L 15 10",
        "M 0,0 L 1,1 Q 1,1 1,1 L 2,3",
    ],
)
def test_empty_segments(svg_d: str):
    res = list(wires_from_svg_path(svg_d))
    assert len(res) == 1
    assert isinstance(res[0], TopoDS_Wire)


def test_empty_paths():
    assert not list(edges_from_svg_path(""))
    assert not list(wires_from_svg_path(""))
    assert not list(faces_from_svg_path(""))


@pytest.mark.parametrize(
    "svg_d, expected_count",
    [
        ("M 0,0", 0),
        ("M 0,0 v 1 h 1 z", 1),
        ("M 0,0 v 1 h 1", 1),
        ("M 10 80 Q 95 10 180 80", 1),
        ("M 10 80 C 40 10, 65 10, 95 80 S 150 150, 180 80", 1),
    ],
)
def test_faces_from_svg_path(svg_d: str, expected_count: int):
    res = list(faces_from_svg_path(svg_d))
    assert len(res) == expected_count
    assert all(isinstance(x, TopoDS_Face) for x in res)


@pytest.mark.parametrize(
    "svg_d, expected_count",
    [
        ("M 0,0 v 1 h 1", 1),
        ("M 0,0 v 1 M 1,0 v 2", 2),
        ("M 0,0 v 1 h 1 z M 2,0 v 1 h 1", 2),
        ("M 0,0 v 1 M 1,0 v 2", 2),
    ],
)
def test_wires_from_svg_path(svg_d: str, expected_count: int):
    res = list(wires_from_svg_path(svg_d))
    assert len(res) == expected_count
    assert all(isinstance(x, TopoDS_Wire) for x in res)


@pytest.mark.parametrize("count", range(3, 10))
def test_concentric_path_nesting(count: int):
    n = count // 2
    if count % 2:
        expected_hole_counts = [0] + [1] * n
    else:
        expected_hole_counts = [1] * n
    res = list(faces_from_svg_path(nested_squares_path(count)))
    assert hole_counts(res) == expected_hole_counts


def test_path_nesting():
    """This path:
    ```
        .-----------------------------------.
        |   .----------------------------.  |
        |   | .--------------.  .-----.  |  |
        |   | | .---.        |  `-----'  |  |
        |   | | |   |  .---. |  .-----.  |  |
        |   | | `---'  `---' |  |     |  |  |
        |   | `--------------'  `-----'  |  |
        |   `----------------------------'  |
        `-----------------------------------'
    ```
    should translate to these faces (order irrelevant):
    ```
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        AAAAA BBBBBBBBBBBBBBBB  CCCCCCC  AAAA
        AAAAA BBB   BBBBBBBBBB  CCCCCCC  AAAA
        AAAAA BBB   BBB     BB  DDDDDDD  AAAA
        AAAAA BBB   BBB     BB  DDDDDDD  AAAA
        AAAAA BBBBBBBBBBBBBBBB  DDDDDDD  AAAA
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    ```
    """
    res = list(
        faces_from_svg_path(
            "M 1,4 L 16,4 L 16,13 L 1,13 L 1,4 Z"
            "M 2,5 L 15,5 L 15,12 L 2,12 L 2,5 Z"
            "M 11,8 L 14,8 L 14,11 L 11,11 L 11,8 Z"
            "M 11,6 L 14,6 L 14,7 L 11,7 L 11,6 Z"
            "M 3,6 L 10,6 L 10,11 L 3,11 L 3,6 Z"
            "M 7,8 L 9,8 L 9,10 L 7,10 L 7,8 Z"
            "M 4,7 L 6,7 L 6,10 L 4,10 L 4,7 Z"
        )
    )

    expected_hole_counts = [0, 0, 1, 2]
    assert hole_counts(res) == expected_hole_counts


def test_svg_doc_path():
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <path d = "M 1,4 L 16,4 L 16,13 L 1,13 L 1,4 Z"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf))
    assert len(imported) == 1


def test_svg_doc_shapes():
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <rect width="300" height="100" fill="none"/>
    <ellipse cx="200" cy="80" rx="100" ry="50" fill="red"/>
    <circle cx="50" cy="100" r="64" fill="red"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf))
    assert len(imported) == 3


@pytest.mark.parametrize("flip_y", [False, True])
def test_svg_doc_orientation(flip_y: bool):
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <rect width="200" height="100"/>
    </svg>
    """
    buf = StringIO(svg_src)

    imported = list(import_svg_document(buf, flip_y=flip_y))

    assert len(imported) == 1
    for face in imported:
        assert isinstance(face, TopoDS_Face)
        assert as_triple(face_normal(face)) == approx((0, 0, 1))


@pytest.mark.parametrize("flip_y", [False, True])
def test_svg_doc_transforms(flip_y: bool):
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <rect width="30" height="20" x="5" y ="15"/>
    <rect width="30" height="20" transform="translate(5 15)"/>
    <g transform="translate(5 15)">
      <rect width="30" height="20"/>
    </g>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf, flip_y=flip_y))
    assert len(imported) == 3

    def assert_bounds(s1: TopoDS_Shape, s2: TopoDS_Shape):
        b1 = bounding_box(s1)
        b2 = bounding_box(s2)
        assert as_triple(b1.CornerMin()) == approx(as_triple(b2.CornerMin()))
        assert as_triple(b1.CornerMax()) == approx(as_triple(b2.CornerMax()))

    first, *others = imported
    for other in others:
        assert_bounds(first, other)


def test_svg_doc_visibility_hidden():
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <g visibility="hidden">
    <rect width="300" height="100"/>
    </g>
    <rect width="300" height="100" display="none"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf))
    assert len(imported) == 0


def test_svg_doc_ignore_visibility():
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <g visibility="hidden">
    <rect width="300" height="100"/>
    </g>
    <rect width="300" height="100" display="none"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf, ignore_visibility=True))
    assert len(imported) == 2


@pytest.mark.parametrize("flip_y", [False, True])
def test_svg_doc_dimensions(flip_y: bool):
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg"
        width="20cm"
        height="10cm"
        viewBox="-100 -100 800 400">
    <rect width="200" height="100"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf, flip_y=flip_y))
    assert len(imported) == 1

    bb = bounding_box(imported[0])
    size = gp_Vec(bb.CornerMin(), bb.CornerMax())
    assert as_triple(size) == approx((50, 25, 0))


def test_svg_doc_metadata():
    svg_src = """
    <svg xmlns="http://www.w3.org/2000/svg">
    <rect width="5" height="10" class="blue" fill="none" stroke="#0000ff"/>
    <rect width="8" height="4" class="red" fill="#ff0000"/>
    <rect width="12" height="3"/>
    </svg>
    """
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf, metadata=ColorAndLabel.Label_by("class")))
    assert len(imported) == 3
    assert imported[0][1].color == (0, 0, 1, 1) and imported[0][1].label == "blue"
    assert imported[1][1].color == (1, 0, 0, 1) and imported[1][1].label == "red"
    assert imported[2][1].color == (0, 0, 0, 1) and imported[2][1].label == ""


def test_svg_nested_use_metadata():
    svg_src = """
    <svg id="svg1" viewBox="-10 -10 35 30" version='1.1' xmlns='http://www.w3.org/2000/svg'>
    <defs>
    <circle id="circle" cx="0" cy="0" r="5" style="opacity:1" />
    <g id="two-circles">
        <use href="#circle" id="blue_fill" transform="translate(-6 0)" fill="blue"/>
        <use href="#circle" id="red_fill" transform="translate(+6 0)" fill="red"/>
    </g>
    </defs>
    <g id="main">
    <use href="#two-circles" id="white_stroke" transform="translate(5,0)" stroke="white"/>
    <use href="#two-circles" id="black_stroke" transform="translate(11,10)" stroke="black"/>
    </g>
    </svg>"""
    buf = StringIO(svg_src)
    imported = list(import_svg_document(buf, metadata=ColorAndLabel.Label_by("id")))

    expected = (
        (("svg1", "main", "white_stroke", "two-circles", "blue_fill"), "circle"),
        (("svg1", "main", "white_stroke", "two-circles", "red_fill"), "circle"),
        (("svg1", "main", "black_stroke", "two-circles", "blue_fill"), "circle"),
        (("svg1", "main", "black_stroke", "two-circles", "red_fill"), "circle"),
    )
    assert len(imported) == len(expected)
    for (_, metadata), (parent_labels, label) in zip(imported, expected):
        assert metadata.label == label
        assert metadata.parent_labels == parent_labels


def nested_squares_path(count: int, x: float = 0, y: float = 0):
    def parts():
        for s in range(1, count + 1):
            yield f"M{x-s},{y-s} H{x+s} V{y+s} H{x-s} Z"

    return " ".join(parts())


def hole_counts(maybe_faces: list[TopoDS_Face]):
    return sorted(
        len(face_inner_wires(face))
        for face in maybe_faces
        if isinstance(face, TopoDS_Face)
    )


def svg_path_tokens(path: Union[SvgPath, str]):
    def split_floats(text: str):
        i = 0
        for m in re.finditer(r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)", text):
            before = text[i : m.start(1)].strip().strip(",")
            if before:
                yield before.strip()
            yield float(m.group(1))
            i = m.end()
        end = text[i:].strip()
        if end:
            yield end

    return list(split_floats(str(path)))
