from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from xml.etree import ElementTree as ET

from OCP.Bnd import Bnd_Box
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape, HLRBRep_TypeOfResultingEdge
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Shape

from .ocp import bounding_box, topoDS_iterator
from .svg import edge_to_svg_path, float_formatter, format_svg_path

HlrEdgeTypeName = Literal["undefined", "isoline", "sewn", "smooth", "sharp", "outline"]

EDGE_TYPES_NAMES: dict[HLRBRep_TypeOfResultingEdge, HlrEdgeTypeName] = {
    HLRBRep_TypeOfResultingEdge.HLRBRep_Undefined: "undefined",
    HLRBRep_TypeOfResultingEdge.HLRBRep_IsoLine: "isoline",
    HLRBRep_TypeOfResultingEdge.HLRBRep_RgNLine: "sewn",
    HLRBRep_TypeOfResultingEdge.HLRBRep_Rg1Line: "smooth",
    HLRBRep_TypeOfResultingEdge.HLRBRep_Sharp: "sharp",
    HLRBRep_TypeOfResultingEdge.HLRBRep_OutLine: "outline",
}


Padding = Union[
    float,
    tuple[float],
    tuple[float, float],
    tuple[float, float, float],
    tuple[float, float, float, float],
]
CssStyle = Mapping[str, Mapping[str, Any]]
RGB = tuple[float, float, float]
EdgeColor = Union[RGB, tuple[RGB, RGB]]


PRIMARY_EDGES = "sharp", "outline"
SECONDARY_EDGES = "undefined", "isoline", "sewn", "smooth"


@dataclass
class HlrEdge:
    type: HLRBRep_TypeOfResultingEdge
    is_hidden: bool
    projected_edge: TopoDS_Edge
    edge_in_3d: TopoDS_Edge
    shape_index: int

    @property
    def type_name(self) -> HlrEdgeTypeName:
        return EDGE_TYPES_NAMES.get(self.type, "undefined")


@dataclass
class HiddenLineRenderer:
    projector: HLRAlgo_Projector

    EDGE_TYPES = [
        # ignore HLRBRep_TypeOfResultingEdge.HLRBRep_Undefined
        HLRBRep_TypeOfResultingEdge.HLRBRep_IsoLine,
        HLRBRep_TypeOfResultingEdge.HLRBRep_RgNLine,
        HLRBRep_TypeOfResultingEdge.HLRBRep_Rg1Line,
        HLRBRep_TypeOfResultingEdge.HLRBRep_Sharp,
        HLRBRep_TypeOfResultingEdge.HLRBRep_OutLine,
    ]

    @classmethod
    def Orthographic(
        cls,
        camera_direction: gp_Dir = gp_Dir(-1, +1, -1),
        camera_up: gp_Dir = gp_Dir(0, 0, 1),
    ):
        camera_ax = gp_Ax2(gp_Pnt(0, 0, 0), camera_direction.Reversed())
        camera_ax.SetYDirection(camera_up)
        projector = HLRAlgo_Projector(camera_ax)
        return cls(projector)

    @classmethod
    def Perspective(
        cls,
        camera_position: gp_Pnt,
        camera_focus: gp_Pnt = gp_Pnt(0, 0, 0),
        focal_length: float = 400,
        camera_up: gp_Dir = gp_Dir(0, 0, 1),
    ):
        camera_ax = gp_Ax2(camera_focus, gp_Dir(gp_Vec(camera_focus, camera_position)))
        camera_ax.SetYDirection(camera_up)
        projector = HLRAlgo_Projector(camera_ax, focal_length)
        return cls(projector)

    def __call__(
        self, shapes: Iterable[TopoDS_Shape], with_hidden: bool = True
    ) -> HiddenLineRender:
        def rough_z(e: HlrEdge):
            trsf = self.projector.FullTransformation()
            pseudo_projected = BRepBuilderAPI_Transform(e.edge_in_3d, trsf).Shape()
            return bounding_box(pseudo_projected).CornerMax().Z()

        return HiddenLineRender(
            sorted(self.compute_edges(shapes, with_hidden=with_hidden), key=rough_z),
            self.projector,
        )

    def compute_edges(self, shapes: Iterable[TopoDS_Shape], with_hidden: bool = True):
        hlr = HLRBRep_Algo()
        for shape in shapes:
            hlr.Add(shape)

        hlr.Projector(self.projector)
        hlr.Update()
        hlr.Hide()

        visibilities = (False, True) if with_hidden else (True,)

        for shape_index in range(hlr.NbShapes()):
            hlr.Select(shape_index + 1)
            hlr_shapes = HLRBRep_HLRToShape(hlr)

            for is_visible, type in product(visibilities, self.EDGE_TYPES):
                edges_in_2d = hlr_shapes.CompoundOfEdges(type, is_visible, In3d=False)
                edges_in_3d = hlr_shapes.CompoundOfEdges(type, is_visible, In3d=True)

                for edge_in_2d, edge_in_3d in zip(
                    topoDS_iterator(edges_in_2d), topoDS_iterator(edges_in_3d)
                ):
                    yield HlrEdge(
                        type=type,
                        is_hidden=not is_visible,
                        projected_edge=TopoDS.Edge_s(edge_in_2d),
                        edge_in_3d=TopoDS.Edge_s(edge_in_3d),
                        shape_index=shape_index,
                    )


class HiddenLineRender(Iterable[HlrEdge]):
    def __init__(
        self, edges: Iterable[HlrEdge], projector: Optional[HLRAlgo_Projector] = None
    ) -> None:
        self.edges = list(edges)
        self.projector = projector

    def __iter__(self) -> Iterator[HlrEdge]:
        return iter(self.edges)

    def bounds(self) -> Bnd_Box:
        return bounding_box(e.projected_edge for e in self)

    def to_svg(
        self,
        width: Optional[float] = 512,
        height: Optional[float] = None,
        *,
        padding: Padding = 16,
        css_style: Optional[CssStyle] = None,
        background: Optional[bool] = None,
        tolerance: float = 1e-6,
        decimals: Optional[int] = None,
    ) -> ET.ElementTree:
        (x0, y0, x1, y1), scale, (W, H) = _viewbox_and_scale(
            self.bounds(),
            width=width,
            height=height,
            padding=padding,
        )
        ty = -(y1 + y0) / scale

        fmt = float_formatter(decimals)

        svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg")
        svg.attrib["viewBox"] = f"{fmt(x0)} {fmt(y0)} {fmt(x1-x0)} {fmt(y1-y0)}"
        if width:
            svg.attrib["width"] = f"{width}"
        if height:
            svg.attrib["height"] = f"{height}"

        if css_style is None:
            css_style = basic_style()
        if css_style:
            ET.SubElement(svg, "style").text = "\n".join(css_style_to_lines(css_style))

        if background is None:
            background = "#background" in css_style
        if background:
            pad = 1
            ET.SubElement(
                svg,
                "rect",
                id="background",
                x=f"{ (x0+x1)/2 - W/2 - pad }",
                y=f"{ (y0+y1)/2 - H/2 - pad }",
                width=f"{ W + 2*pad  }",
                height=f"{ H + 2*pad  }",
            )

        # can't scale/mirror `BRep_CurveOnSurface` edges :( so we do that in the SVG
        # and use `vector-effect: non-scaling-stroke` style to keep intended line witdhs
        transformed_group = ET.SubElement(
            svg,
            "g",
            transform=f"scale({fmt(scale)} {fmt(-scale)}) translate(0 {fmt(ty)})",
        )

        for i, edge in enumerate(self):
            classnames = [f"s{edge.shape_index}", edge.type_name]
            if edge.is_hidden:
                classnames.append("hidden")

            attrs = {
                "id": f"e{i}",
                "d": format_svg_path(
                    edge_to_svg_path(edge.projected_edge, tolerance=tolerance),
                    decimals=decimals,
                ),
                "class": " ".join(classnames),
            }
            ET.SubElement(transformed_group, "path", attrs)

        return ET.ElementTree(svg)


def write_svg(
    svg: ET.ElementTree,
    f: Union[str, Path, BinaryIO],
    indent: bool = True,
):
    if indent:
        ET.indent(svg)
    svg.write(f, xml_declaration=True, encoding="utf-8")


def basic_style(
    color: EdgeColor = ((0, 0, 0), (0.3, 0.3, 0.3)),
    linewidth: float = 1,
    secondary_linewidth: Optional[float] = None,
    background_color: Optional[RGB] = None,
    shape_colors: Optional[Mapping[int, EdgeColor]] = None,
) -> CssStyle:
    if secondary_linewidth is None:
        secondary_linewidth = linewidth * 0.75
    color, hidden_color = _edge_color_pair(color)

    style = {
        "path": {
            "fill": "none",
            "stroke": hexcolor(color),
            "stroke-width": linewidth,
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
            "vector-effect": "non-scaling-stroke",
        },
        ", ".join(f".{t}" for t in SECONDARY_EDGES): {
            "stroke-width": secondary_linewidth,
        },
        ".hidden": {
            "stroke": hexcolor(hidden_color),
            "stroke-dasharray": f"{1.5*linewidth} {2.5*linewidth}",
        },
    }

    if background_color:
        style["#background"] = {"fill": hexcolor(background_color)}

    if shape_colors:
        for i, color in shape_colors.items():
            color, hidden_color = _edge_color_pair(color)
            style[f".s{i}"] = {"stroke": hexcolor(color)}
            style[f".s{i}.hidden"] = {"stroke": hexcolor(hidden_color)}

    return style


def hexcolor(rgb: tuple[float, float, float]):
    r, g, b = [min(max(int(round(v * 255)), 0), 255) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def _edge_color_pair(rgb_or_pair: EdgeColor):
    if len(rgb_or_pair) == 2:
        return rgb_or_pair
    else:
        return rgb_or_pair, rgb_or_pair


def _viewbox_and_scale(
    bbox: Bnd_Box,
    width: Optional[float] = None,
    height: Optional[float] = None,
    *,
    padding: Padding = 0,
):
    padding_t, padding_r, padding_b, padding_l = _normalize_padding(padding)

    c0 = bbox.CornerMin()
    c1 = bbox.CornerMax()
    bounds_width = c1.X() - c0.X()
    bounds_height = c1.Y() - c0.Y()

    if width and height:
        scale = min(
            (width - padding_r - padding_l) / bounds_width,
            (height - padding_b - padding_t) / bounds_height,
        )
    elif width:
        scale = (width - padding_r - padding_l) / bounds_width
        height = bounds_height * scale + padding_b + padding_t
    elif height:
        scale = (height - padding_b - padding_t) / bounds_height
        width = bounds_width * scale + padding_r + padding_l
    else:
        scale = 1
        width = bounds_width + padding_r + padding_l
        height = bounds_height + padding_b + padding_t

    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(), scale)
    c0.Transform(trsf)
    c1.Transform(trsf)

    x0 = c0.X() - padding_l
    y0 = c0.Y() - padding_b
    x1 = c1.X() + padding_r
    y1 = c1.Y() + padding_t

    return (x0, y0, x1, y1), scale, (width, height)


def _normalize_padding(p: Padding) -> tuple[float, float, float, float]:
    if isinstance(p, Sequence):
        if len(p) == 1:
            return p[0], p[0], p[0], p[0]
        elif len(p) == 2:
            return p[0], p[1], p[0], p[1]
        elif len(p) == 3:
            return p[0], p[1], p[2], p[1]
        else:
            return p[0], p[1], p[2], p[3]
    else:
        return p, p, p, p


def css_style_to_lines(style: CssStyle):
    for selector, props in style.items():
        yield selector + " {"
        for prop, value in props.items():
            yield f"{prop}: { value };"
        yield "}"
