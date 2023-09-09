from io import BytesIO
from itertools import groupby
from math import radians
from typing import Optional

import pytest
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.TopoDS import TopoDS_Shape

from ocpsvg.hlr import HiddenLineRenderer, Padding, basic_style, write_svg
from ocpsvg.ocp import edge_from_curve, segment_curve


def test_edge_order_ortho():
    S0 = cylinder(1, 3, x=0, y=0)
    S1 = cylinder(1, 3, x=4, y=4)
    S2 = cylinder(1, 3, x=8, y=8)

    render = HiddenLineRenderer.Orthographic(gp_Dir(-1, -1, -1))([S0, S1, S2])
    assert [i for i, _ in groupby(e.shape_index for e in render)] == [0, 1, 2]

    render = HiddenLineRenderer.Orthographic(gp_Dir(+1, +1, -1))([S0, S1, S2])
    assert [i for i, _ in groupby(e.shape_index for e in render)] == [2, 1, 0]


def test_edge_order_perspective():
    S0 = cylinder(1, 3, x=0, y=0)
    S1 = cylinder(1, 3, x=4, y=4)
    S2 = cylinder(1, 3, x=8, y=8)

    render = HiddenLineRenderer.Perspective(gp_Pnt(+1, +1, +1))([S0, S1, S2])
    assert [i for i, _ in groupby(e.shape_index for e in render)] == [0, 1, 2]

    render = HiddenLineRenderer.Perspective(gp_Pnt(-1, -1, +1))([S0, S1, S2])
    assert [i for i, _ in groupby(e.shape_index for e in render)] == [2, 1, 0]


def test_to_svg():
    shapes = [
        cylinder(1, 3, x=0, y=0),
        cylinder(1, 3, x=4, y=4),
        cylinder(1, 3, x=8, y=8),
    ]

    render = HiddenLineRenderer.Orthographic(gp_Dir(-1, -1, -1))(shapes)
    svg = render.to_svg()
    path_elements = svg.findall("*/path")

    assert len(path_elements) == len(list(render))

    for element, edge in zip(path_elements, render):
        assert f"s{edge.shape_index}" in element.attrib["class"]


def test_to_svg_background():
    shapes = [
        cylinder(1, 3, x=0, y=0),
    ]
    render = HiddenLineRenderer.Orthographic(gp_Dir(-1, -1, -1))(shapes)

    svg = render.to_svg(css_style={})
    assert not svg.findall('rect[@id="background"]')

    svg = render.to_svg(background=True)
    assert svg.findall('rect[@id="background"]')

    svg = render.to_svg(css_style={"#background": {"color": "white"}})
    assert svg.findall('rect[@id="background"]')


def test_basic_style_color():
    css = basic_style(color=((1, 0, 0), (0, 1, 0)))
    assert css["path"]["stroke"] == "#ff0000"
    assert css[".hidden"]["stroke"] == "#00ff00"

    css = basic_style(color=(0, 0, 0))
    assert css["path"]["stroke"] == "#000000"
    assert css[".hidden"]["stroke"] == "#000000"


def test_basic_style_shape_color():
    css = basic_style(
        shape_colors={
            0: ((1, 0, 0), (0, 1, 0)),
            2: ((1, 0, 0), (0, 0, 1)),
        }
    )
    assert css[".s0"]["stroke"] == "#ff0000"
    assert css[".s0.hidden"]["stroke"] == "#00ff00"
    assert css[".s2"]["stroke"] == "#ff0000"
    assert css[".s2.hidden"]["stroke"] == "#0000ff"


def test_basic_style_background_color():
    css = basic_style(background_color=None)
    assert "#background" not in css

    css = basic_style(background_color=(1, 1, 1))
    assert css["#background"]["fill"] == "#ffffff"


# def test_viewport():
#     shapes = [edge_from_curve(segment_curve(gp_Pnt(1, 2, 0), gp_Pnt(11, 22, 3)))]
#     render = HiddenLineRenderer.Orthographic(gp_Dir(-1, 0, 0))(shapes)
#     svg = render.to_svg(width=10, padding=2, css_style={})


def test_write_svg():
    shapes = [edge_from_curve(segment_curve(gp_Pnt(0, 0, 0), gp_Pnt(1, 2, 3)))]
    render = HiddenLineRenderer.Orthographic(gp_Dir(-1, 0, 0))(shapes)

    svg = render.to_svg(width=10, padding=2, css_style={})
    buf = BytesIO()
    write_svg(svg, buf)
    assert buf.getvalue() == (
        b"<?xml version='1.0' encoding='utf-8'?>\n<svg xmlns=\"http://www.w3.org"
        b'/2000/svg" viewBox="-2 -2 10 13" width="10">\n  <g transform="scale(3 -3)'
        b' translate(0 -3)">\n    <path id="e0" d="M0,0 L2,3" class="s0 sharp" />\n '
        b" </g>\n</svg>"
    )


@pytest.mark.parametrize(
    "params, expected",
    [
        ((None, None, 0), ("2 3 10 10", None, None)),
        ((200, None, 0), ("40 60 200 200", "200", None)),
        ((None, 200, 0), ("40 60 200 200", None, "200")),
        ((400, 200, 0), ("40 60 200 200", "400", "200")),
        ((200, 400, 0), ("40 60 200 200", "200", "400")),
        ((200, None, 10), ("26 44 200 200", "200", None)),
        ((200, None, (20,)), ("12 28 200 200", "200", None)),
        ((200, None, (10, 20)), ("12 38 200 180", "200", None)),
        ((200, None, (10, 20, 30)), ("12 18 200 200", "200", None)),
        ((200, None, (10, 20, 30, 40)), ("-12 12 200 180", "200", None)),
    ],
)
def test_size_and_padding(
    params: tuple[Optional[float], Optional[float], Padding],
    expected: tuple[str, Optional[str], Optional[str]],
):
    shapes = [edge_from_curve(segment_curve(gp_Pnt(1, 2, 3), gp_Pnt(11, 12, 13)))]
    render = HiddenLineRenderer.Orthographic(gp_Dir(-1, 0, 0))(shapes)
    width, height, padding = params
    svg = render.to_svg(width=width, height=height, padding=padding)
    viewbox = svg.getroot().attrib["viewBox"]
    width = svg.getroot().attrib.get("width")
    height = svg.getroot().attrib.get("height")
    assert (viewbox, width, height) == expected


def cylinder(
    radius: float, height: float, x: float = 0, y: float = 0, z: float = 0
) -> TopoDS_Shape:
    ax = gp_Ax2(gp_Pnt(x, y, z), gp_Dir(0, 0, 1))
    return BRepPrimAPI_MakeCylinder(ax, radius, height, radians(360)).Solid()
