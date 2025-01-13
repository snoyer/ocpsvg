import logging
import pathlib
import warnings
from itertools import chain
from math import degrees, pi
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    Union,
    overload,
)

import svgelements
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.Geom import Geom_BezierCurve
from OCP.GeomAbs import GeomAbs_CurveType
from OCP.GeomAdaptor import GeomAdaptor_Curve
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.StdFail import StdFail_NotDone
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Wire

from .ocp import (
    CurveOrAdaptor,
    InvalidWiresForFace,
    bezier_curve,
    circle_curve,
    curve_and_adaptor,
    curve_to_beziers,
    curve_to_polyline,
    edge_from_curve,
    edge_to_curve,
    ellipse_curve,
    faces_from_wire_soup,
    is_reversed,
    segment_curve,
    topoDS_iterator,
    wire_explorer,
    wire_from_continuous_edges,
)

__all__ = [
    "import_svg_document",
    "DocumentInfo",
    "ColorAndLabel",
    "faces_from_svg_path",
    "wires_from_svg_path",
    "edges_from_svg_path",
    "continuous_edges_from_svg_path",
    "face_to_svg_path",
    "wire_to_svg_path",
    "edge_to_svg_path",
    "curve_to_svg_path",
    "SvgPathCommand",
    "format_svg_path",
]


flip_y = gp_Trsf()
flip_y.SetMirror(gp_Ax1(gp_Pnt(), gp_Dir(1, 0, 0)))


logger = logging.getLogger(__name__)

SvgPathCommand = Union[
    tuple[Literal["M"], float, float],
    tuple[Literal["L"], float, float],
    tuple[Literal["Q"], float, float, float, float],
    tuple[Literal["C"], float, float, float, float, float, float],
    tuple[Literal["A"], float, float, float, int, int, float, float],
    tuple[Literal["Z"]],
]


class float_formatter:
    """float formatter with given precision and no trailing zeroes."""

    def __init__(self, decimals: Optional[int] = None):
        self.format_spec = f".{decimals or 12}f"

    def __call__(self, v: float):
        return v.__format__(self.format_spec).rstrip("0").rstrip(".")


def format_svg_path(
    path: Iterable[SvgPathCommand], *, decimals: Optional[int] = None
) -> str:
    ff = float_formatter(decimals)
    return " ".join(f"{cmd[0]}{','.join(map(ff, cmd[1:]))}" for cmd in path)


def format_svg(path: Iterable[SvgPathCommand], float_format: str = "f") -> str:
    warnings.warn(
        "`format_svg` is deprecated, use `format_svg_path` instead.",
        FutureWarning,
        stacklevel=2,
    )
    return " ".join(
        f"{cmd[0]}{','.join(arg.__format__(float_format) for arg in cmd[1:])}"
        for cmd in path
    )


SvgPathLike = Union[str, Iterable[SvgPathCommand], svgelements.Path]
ShapeElement = svgelements.Shape
ParentElement = Union[svgelements.Group, svgelements.Use]

FaceOrWire = Union[TopoDS_Wire, TopoDS_Face]


class DocumentInfo(NamedTuple):
    width: float
    height: float


T = TypeVar("T")


class ItemsFromDocument(Iterable[T]):
    def __init__(self, elements: Iterator[T], doc_info: DocumentInfo) -> None:
        self.elements = elements
        self.doc_info = doc_info

    def __iter__(self) -> Iterator[T]:
        return self.elements


M = TypeVar("M")


@overload
def import_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    *,
    flip_y: bool = True,
    ignore_visibility: bool = False,
    metadata: Optional[Callable[[ShapeElement, Sequence[ParentElement]], M]],
) -> ItemsFromDocument[tuple[FaceOrWire, M]]: ...


@overload
def import_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    *,
    flip_y: bool = True,
    ignore_visibility: bool = False,
) -> ItemsFromDocument[FaceOrWire]: ...


def import_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    *,
    flip_y: bool = True,
    ignore_visibility: bool = False,
    metadata: Optional[Callable[[ShapeElement, Sequence[ParentElement]], M]] = None,
) -> Union[ItemsFromDocument[tuple[FaceOrWire, M]], ItemsFromDocument[FaceOrWire]]:
    """Import shapes from an SVG document as faces and/or wires.

    Each visible shape and path is converted to zero or more Face if it is filled,
    and to zero or more Wire if it is not filled.

    This importer does not cover the whole SVG specification,
    its most notable known limitations are:

    - degenerate and self-crossing paths may result in invalid faces and wires
    - clipping, both by clipping paths and viewport, is ignored
    - graphic properties such as line strokes and pattern fills are ignored

    Documents relying on these features need to be pre-processed externally.

    :param svg_file: input SVG document
    :param flip_y: whether to mirror the Y-coordinates
        to compensate for SVG's top left origin, defaults to True
    :param ignore_visibility: whether to ignore visibility
        attribute and process hidden elements.
    :param metadata: funtion to generate metadata from the source SVG element
    :raises IOError:
    :raises SyntaxError:
    :raises ValueError:
    """

    def doc_transform(info: DocumentInfo):
        mirror = gp_Trsf()
        mirror.SetMirror(gp_Ax1(gp_Pnt(0, info.height / 2, 0), gp_Dir(1, 0, 0)))

        def f_flip(shape: FaceOrWire) -> FaceOrWire:
            mirrored = BRepBuilderAPI_Transform(shape, mirror, False, False).Shape()
            mirrored.Reverse()
            if isinstance(shape, TopoDS_Face):
                return TopoDS.Face_s(mirrored)
            elif isinstance(shape, TopoDS_Wire):
                return TopoDS.Wire_s(mirrored)
            else:
                raise AssertionError(f"somehow got unexpected shape {shape}")

        def f_identity(shape: FaceOrWire) -> FaceOrWire:
            return shape

        return f_flip if flip_y else f_identity

    def process_wire(
        wires: list[TopoDS_Wire],
        is_filled: bool,
    ) -> Iterator[FaceOrWire]:
        if is_filled:
            try:
                yield from faces_from_wire_soup(wires)
            except InvalidWiresForFace:
                logger.warning("filled shape could not be converted to face")  # TODO
                yield from wires
        else:
            yield from wires

    if metadata:
        wires_from_doc = wires_from_svg_document(
            svg_file,
            ignore_visibility=ignore_visibility,
            metadata_factory=metadata,
        )
        transform = doc_transform(wires_from_doc.doc_info)
        items = (
            (transform(face_or_wire), m)
            for wires, is_filled, m in wires_from_doc
            for face_or_wire in process_wire(wires, is_filled)
        )
        return ItemsFromDocument(items, wires_from_doc.doc_info)
    else:
        wires_from_doc = wires_from_svg_document(
            svg_file,
            ignore_visibility=ignore_visibility,
            metadata_factory=None,
        )
        transform = doc_transform(wires_from_doc.doc_info)
        items = (
            transform(face_or_wire)
            for wires, is_filled, _metadata in wires_from_doc
            for face_or_wire in process_wire(wires, is_filled)
        )
        return ItemsFromDocument(items, wires_from_doc.doc_info)


class ColorAndLabel:
    def __init__(
        self,
        element: ShapeElement,
        parents: Sequence[ParentElement],
        label_by: str = "id",
    ) -> None:
        self.fill_color = self._color(element.fill)
        self.stroke_color = self._color(element.stroke) or (0, 0, 0, 1)
        self.label = self._label(element, label_by)
        self.parent_labels = tuple(self._label(parent, label_by) for parent in parents)

    def color_for(self, shape: TopoDS_Shape):
        """Fill color if shape should be filled stroke color otherwise."""
        filled = not isinstance(shape, (TopoDS_Wire, TopoDS_Edge))
        return self.fill_color if filled and self.fill_color else self.stroke_color

    @staticmethod
    def _color(
        color: Union[svgelements.Color, None]
    ) -> Union[tuple[float, float, float, float], None]:
        if color and color.value:  # type: ignore
            try:
                rgba = color.red, color.green, color.blue, color.alpha  # type: ignore
                return tuple(float(v) / 255 for v in rgba)  # type: ignore
            except TypeError:
                return 0, 0, 0, 1

    @staticmethod
    def _label(element: Union[ShapeElement, ParentElement], label_by: str):
        try:
            return str(element.values[label_by])  # type: ignore
        except (KeyError, AttributeError):
            return ""

    @classmethod
    def Label_by(cls, label_by: str = "id"):
        def f(element: ShapeElement, parents: Sequence[ParentElement]):
            return cls(element, parents, label_by=label_by)

        return f


####


def faces_from_svg_path(path: SvgPathLike) -> Iterator[TopoDS_Face]:
    """Create faces from an SVG path.

    :param SvgPathLike path: input SVG path
    :yield: faces
    :raises SyntaxError:
    :raises ValueError:
    """
    return faces_from_wire_soup(wires_from_svg_path(path))


def wires_from_svg_element(element: ShapeElement) -> Iterator[TopoDS_Wire]:
    def check_unskewed_transform() -> Union[tuple[float, float, float], None]:
        o = svgelements.Point(0, 0) * element.transform
        x = svgelements.Point(1, 0) * element.transform
        y = svgelements.Point(0, 1) * element.transform
        is_skewed = abs(o.angle_to(y) - o.angle_to(x)) != pi / 2  # type: ignore
        if not is_skewed:
            return o.distance_to(x), o.distance_to(y), o.angle_to(x)  # type: ignore

    element.reify()

    if isinstance(element, (svgelements.Circle, svgelements.Ellipse)) and (
        scale_and_angle := check_unskewed_transform()
    ):
        cx = float(element.cx)  # type: ignore
        cy = float(element.cy)  # type: ignore
        origin = svgelements.Point(cx, cy) * element.transform
        center = gp_Pnt(origin.real, origin.imag, 0)  # type: ignore
        scale_x, scale_y, angle = scale_and_angle
        r1 = float(element.rx) * scale_x  # type: ignore
        r2 = float(element.ry) * scale_y  # type: ignore
        if r1 == r2:
            curve = circle_curve(r1, center=center)
        else:
            curve = ellipse_curve(r1, r2, center=center, rotation=degrees(angle))
        yield wire_from_continuous_edges((edge_from_curve(curve),))

    elif path := svg_element_to_path(element):
        yield from wires_from_svg_path(path)


def svg_element_to_path(element: ShapeElement):
    path = svgelements.Path(element)
    if len(path):
        path.reify()
        return path


def wires_from_svg_path(path: SvgPathLike) -> Iterator[TopoDS_Wire]:
    """Create wires from an SVG path.

    :param SvgPathLike path: input SVG path
    :yield: wires
    :raises SyntaxError:
    :raises ValueError:
    """

    for edges, closed in continuous_edges_from_svg_path(path):
        yield wire_from_continuous_edges(edges, closed=closed)


def edges_from_svg_path(path: SvgPathLike) -> Iterator[TopoDS_Edge]:
    """Create edges from an SVG path.

    :param SvgPathLike path: input SVG path
    :yield: edges
    :raises SyntaxError:
    :raises ValueError:
    """
    for edges, _ in continuous_edges_from_svg_path(path):
        yield from edges


def continuous_edges_from_svg_path(
    path: SvgPathLike,
) -> Iterator[tuple[Iterable[TopoDS_Edge], bool]]:
    def p(c: Union[svgelements.Point, None]):
        if c is None:  # pragma: nocover
            logger.warning("found None point in %s", path)
            return gp_Pnt(0, 0, 0)
        else:
            return gp_Pnt(c.real, c.imag, 0)  # type: ignore

    def curves_from_segment(
        segment: Union[
            svgelements.Line,
            svgelements.QuadraticBezier,
            svgelements.CubicBezier,
            svgelements.Arc,
        ]
    ):
        if isinstance(segment, svgelements.Move):
            pass
        elif isinstance(segment, svgelements.Line):
            if segment.start != segment.end:
                return segment_curve(p(segment.start), p(segment.end))
        elif isinstance(segment, svgelements.QuadraticBezier):
            return bezier_curve(p(segment.start), p(segment.control), p(segment.end))
        elif isinstance(segment, svgelements.CubicBezier):
            return bezier_curve(
                p(segment.start),
                p(segment.control1),
                p(segment.control2),
                p(segment.end),
            )
        elif isinstance(segment, svgelements.Arc):
            start_angle = segment.theta
            end_angle = segment.theta + segment.delta
            return ellipse_curve(
                segment.radius.real,  # type: ignore
                segment.radius.imag,  # type: ignore
                start_angle=min(start_angle, end_angle),
                end_angle=max(end_angle, start_angle),
                clockwise=end_angle > start_angle,
                center=p(segment.center),
                rotation=degrees(segment.get_rotation()),
            )
        elif isinstance(segment, svgelements.Close):
            if segment.start != segment.end:
                return segment_curve(p(segment.start), p(segment.end))
        else:  # pragma: nocover
            logger.warning(f"unexpected segment type: {type(segment)}")
            return segment_curve(p(segment.start), p(segment.end))

    def edges_from_path(
        path: svgelements.Path,
    ):
        for segment in path:  # type: ignore
            try:
                if curve := curves_from_segment(segment):  # type: ignore
                    yield edge_from_curve(curve)
            except (StdFail_NotDone, ValueError):  # pragma: nocover
                logger.debug("invalid %s", _SegmentInPath(segment, path))

    path = _path_from_SvgPathLike(path)
    for subpath, closed in _continuous_subpaths(path):
        if edges := list(edges_from_path(subpath)):
            yield edges, closed


class _SegmentInPath:
    def __init__(self, segment: Any, path: svgelements.Path) -> None:
        self.segment = segment
        self.path = path

    def __str__(self) -> str:
        d = self.path.d()  # type: ignore
        return f"`{self.segment}` segment in path `{d}`"


####


def face_to_svg_path(
    face: TopoDS_Face,
    *,
    tolerance: float,
    use_cubics: bool = True,
    use_quadratics: bool = True,
    use_arcs: bool = True,
    split_full_arcs: bool = True,
) -> Iterator[SvgPathCommand]:
    for wire in topoDS_iterator(face):
        cmd = None
        for cmd in wire_to_svg_path(
            TopoDS.Wire_s(wire),
            tolerance=tolerance,
            use_cubics=use_cubics,
            use_quadratics=use_quadratics,
            use_arcs=use_arcs,
            split_full_arcs=split_full_arcs,
            with_first_move=True,
        ):
            yield cmd
        if cmd and cmd != ("Z",):
            yield "Z",


def wire_to_svg_path(
    wire: TopoDS_Wire,
    *,
    tolerance: float,
    use_cubics: bool = True,
    use_quadratics: bool = True,
    use_arcs: bool = True,
    split_full_arcs: bool = True,
    with_first_move: bool = True,
) -> Iterator[SvgPathCommand]:
    # Storage order isn't guaranteed so wo need to use `BRepTools_WireExplorer` for ordering
    # and regular `TopoDS_Iterator` to check that all edges have been visited.
    ordered_edges = list(wire_explorer(wire))

    yield from chain.from_iterable(
        edge_to_svg_path(
            TopoDS.Edge_s(edge),
            tolerance=tolerance,
            use_cubics=use_cubics,
            use_quadratics=use_quadratics,
            use_arcs=use_arcs,
            split_full_arcs=split_full_arcs,
            with_first_move=with_first_move and i == 0,
        )
        for i, edge in enumerate(ordered_edges)
    )

    # In the nominal case of nice wires we should be done here...
    # but wires can be non-manifold or otherwise degenerate and may not have visit them all.
    # We'll add remaining edges individually

    #TODO use a set if/when OCP implements `__eq__`
    all_edges = {hash(e): e for e in map(TopoDS.Edge_s, topoDS_iterator(wire))}

    if len(ordered_edges) < len(all_edges):
        for e in ordered_edges:
            all_edges.pop(hash(e))

        yield from chain.from_iterable(
            edge_to_svg_path(
                TopoDS.Edge_s(edge),
                tolerance=tolerance,
                use_cubics=use_cubics,
                use_quadratics=use_quadratics,
                use_arcs=use_arcs,
                split_full_arcs=split_full_arcs,
                with_first_move=True,
            )
            for edge in all_edges.values()
        )


def edge_to_svg_path(
    edge: TopoDS_Edge,
    *,
    tolerance: float,
    use_cubics: bool = True,
    use_quadratics: bool = True,
    use_arcs: bool = True,
    split_full_arcs: bool = True,
    with_first_move: bool = True,
) -> Iterator[SvgPathCommand]:
    return curve_to_svg_path(
        edge_to_curve(edge),
        tolerance=tolerance,
        use_cubics=use_cubics,
        use_quadratics=use_quadratics,
        use_arcs=use_arcs,
        split_full_arcs=split_full_arcs,
        with_first_move=with_first_move,
        reverse=is_reversed(edge),
    )


def curve_to_svg_path(
    curve_or_adaptor: CurveOrAdaptor,
    *,
    tolerance: float,
    use_cubics: bool = True,
    use_quadratics: bool = True,
    use_arcs: bool = True,
    split_full_arcs: bool = True,
    with_first_move: bool = True,
    reverse: bool = False,
) -> Iterator[SvgPathCommand]:
    _curve, adaptor = curve_and_adaptor(curve_or_adaptor)
    curve_type = adaptor.GetType()

    try:
        if curve_type == GeomAbs_CurveType.GeomAbs_Line:
            p0 = adaptor.Value(adaptor.FirstParameter())
            p1 = adaptor.Value(adaptor.LastParameter())
            if reverse:
                p0, p1 = p1, p0

            if with_first_move:
                yield "M", p0.X(), p0.Y()
            yield "L", p1.X(), p1.Y()

        elif use_arcs and curve_type in (
            GeomAbs_CurveType.GeomAbs_Circle,
            GeomAbs_CurveType.GeomAbs_Ellipse,
        ):
            if curve_type == GeomAbs_CurveType.GeomAbs_Circle:
                circle = adaptor.Circle()
                r1 = r2 = circle.Radius()
                axis = circle.XAxis()
            else:
                ellipse = adaptor.Ellipse()
                r1, r2 = ellipse.MajorRadius(), ellipse.MinorRadius()
                axis = ellipse.XAxis()

            yield from ellipse_to_svg_path(
                adaptor,
                r1,
                r2,
                axis,
                split_full_arcs=split_full_arcs,
                with_first_move=with_first_move,
                reverse=reverse,
            )

        elif use_cubics or use_quadratics:
            beziers = curve_to_beziers(
                adaptor, tolerance=tolerance, max_degree=3 if use_cubics else 2
            )
            if reverse:
                beziers = reversed(list(beziers))
            for i, bezier in enumerate(beziers):
                yield from bezier_to_svg_path(
                    bezier,
                    use_quadratics=use_quadratics,
                    with_first_move=with_first_move and i == 0,
                    reverse=reverse,
                )
        else:
            yield from polyline_to_svg_path(
                curve_to_polyline(adaptor, tolerance=tolerance),
                with_first_move=with_first_move,
                closed=adaptor.IsClosed(),
                reverse=reverse,
            )
    except Exception as e:  # pragma: nocover
        logger.error(
            "failed to convert %s curve, falling back to polyline approximation",
            curve_type,
            exc_info=e,
        )
        yield from polyline_to_svg_path(
            curve_to_polyline(adaptor, tolerance=tolerance),
            with_first_move=with_first_move,
            closed=adaptor.IsClosed(),
            reverse=reverse,
        )


def ellipse_to_svg_path(
    adaptor_curve: Union[GeomAdaptor_Curve, BRepAdaptor_Curve],
    r1: float,
    r2: float,
    axis: gp_Ax1,
    with_first_move: bool = True,
    split_full_arcs: bool = True,
    reverse: bool = False,
) -> Iterator[SvgPathCommand]:
    t0 = adaptor_curve.FirstParameter()
    t1 = adaptor_curve.LastParameter()
    if reverse:
        t0, t1 = t1, t0
    p0 = adaptor_curve.Value(t0)
    pm = adaptor_curve.Value((t1 + t0) / 2.0)
    p1 = adaptor_curve.Value(t1)

    a = gp_Vec(0, 0, 1).DotCross(gp_Vec(pm, p0), gp_Vec(pm, p1))

    angle = axis.Direction().AngleWithRef(gp_Dir(1, 0, 0), gp_Dir(0, 0, -1))
    large_arc = 1 if t1 - t0 > pi else 0
    sweep = 1 if a < 0 else 0

    if with_first_move:
        yield "M", p0.X(), p0.Y()

    if adaptor_curve.IsClosed():
        if split_full_arcs:
            yield "A", r1, r2, 180, 1, 1, pm.X(), pm.Y()
            yield "A", r1, r2, 180, 1, 1, p1.X(), p1.Y()
        else:
            almost_one = 1 - 1e-8  # seem to be a close as we can get for most viewers
            pq = adaptor_curve.Value(t0 + (t1 - t0) * almost_one)
            yield "A", r1, r2, 360 * almost_one, 1, 0, pq.X(), pq.Y()
    else:
        yield "A", r1, r2, degrees(angle), large_arc, sweep, p1.X(), p1.Y()


def bezier_to_svg_path(
    bezier: Geom_BezierCurve,
    *,
    use_quadratics: bool = True,
    with_first_move: bool = True,
    reverse: bool = False,
) -> Iterator[SvgPathCommand]:
    degree = bezier.Degree()

    p0 = bezier.Pole(1)

    if degree == 3:
        p1 = bezier.Pole(2)
        p2 = bezier.Pole(3)
        p3 = bezier.Pole(4)
        if reverse:
            p0, p1, p2, p3 = p3, p2, p1, p0

        if with_first_move:
            yield "M", p0.X(), p0.Y()
        yield "C", p1.X(), p1.Y(), p2.X(), p2.Y(), p3.X(), p3.Y()

    elif degree == 2:
        p1 = bezier.Pole(2)
        p2 = bezier.Pole(3)
        if reverse:
            p0, p1, p2 = p2, p1, p0

        if use_quadratics:
            if with_first_move:
                yield "M", p0.X(), p0.Y()
            yield "Q", p1.X(), p1.Y(), p2.X(), p2.Y()
        else:
            p1b = gp_Vec(p0.X(), p0.Y(), 0).Added(gp_Vec(p0, p1).Multiplied(2 / 3))
            p2b = gp_Vec(p2.X(), p2.Y(), 0).Added(gp_Vec(p2, p1).Multiplied(2 / 3))
            if with_first_move:
                yield "M", p0.X(), p0.Y()
            yield "C", p1b.X(), p1b.Y(), p2b.X(), p2b.Y(), p2.X(), p2.Y()

    elif degree == 1:
        p1 = bezier.Pole(2)
        if reverse:
            p0, p1 = p1, p0

        if with_first_move:
            yield "M", p0.X(), p0.Y()
        yield "L", p1.X(), p1.Y()

    else:
        raise ValueError(f"could not convert {bezier} to SVG path")


def polyline_to_svg_path(
    points: Iterable[gp_Pnt],
    *,
    with_first_move: bool = True,
    closed: bool = False,
    reverse: bool = False,
) -> Iterator[SvgPathCommand]:
    points = list(points)
    if points:
        first, *others = reversed(points) if reverse else points

        if with_first_move:
            yield "M", first.X(), first.Y()
        for point in others:
            yield "L", point.X(), point.Y()

        if closed:
            yield "Z",


####


@overload
def wires_from_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    metadata_factory: Callable[[ShapeElement, Sequence[ParentElement]], M],
    *,
    ignore_visibility: bool = False,
) -> ItemsFromDocument[tuple[list[TopoDS_Wire], bool, M]]: ...


@overload
def wires_from_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    metadata_factory: None,
    *,
    ignore_visibility: bool = False,
) -> ItemsFromDocument[tuple[list[TopoDS_Wire], bool, None]]: ...


def wires_from_svg_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    metadata_factory: Optional[Callable[[ShapeElement, Sequence[ParentElement]], M]],
    *,
    ignore_visibility: bool = False,
) -> Union[
    ItemsFromDocument[tuple[list[TopoDS_Wire], bool, M]],
    ItemsFromDocument[tuple[list[TopoDS_Wire], bool, None]],
]:
    elements = find_shapes_svg_in_document(
        svg_file, ignore_visibility=ignore_visibility
    )

    def is_filled(element: ShapeElement):
        fill = element.fill
        return fill.value is not None  # type: ignore

    if callable(metadata_factory):
        wires = (
            (
                list(wires_from_svg_element(source_element)),
                is_filled(source_element),
                metadata_factory(source_element, source_parents),
            )
            for source_element, source_parents in elements
        )
        return ItemsFromDocument(wires, elements.doc_info)
    else:
        wires = (
            (
                list(wires_from_svg_element(source_element)),
                is_filled(source_element),
                None,
            )
            for source_element, _source_parents in elements
        )
        return ItemsFromDocument(wires, elements.doc_info)


def find_shapes_svg_in_document(
    svg_file: Union[str, pathlib.Path, TextIO],
    *,
    ignore_visibility: bool = False,
) -> ItemsFromDocument[tuple[ShapeElement, tuple[ParentElement, ...]]]:
    def walk_svg_element(
        element: svgelements.SVGElement, parents: tuple[ParentElement, ...] = ()
    ) -> Iterator[tuple[ShapeElement, tuple[ParentElement, ...]]]:
        if isinstance(element, ShapeElement):
            yield element, parents
        elif isinstance(element, (svgelements.Group, svgelements.Use)):
            new_parents = *parents, element
            for child in element:  # type: ignore
                yield from walk_svg_element(child, new_parents)  # type: ignore

    parsed_svg = svgelements.SVG.parse(  # type: ignore
        (
            resolve_path(svg_file)
            if isinstance(svg_file, (str, pathlib.Path))
            else svg_file
        ),
        parse_display_none=ignore_visibility,
        ppi=25.4,  # inches to millimiters
    )

    def elements():
        for element, parents in walk_svg_element(parsed_svg):  # type: ignore
            if not ignore_visibility:
                try:
                    visibility = element.values["visibility"]  # type: ignore
                    if visibility in ("hidden", "collapse"):  # type: ignore
                        continue
                except (KeyError, AttributeError):
                    pass

            if isinstance(element, svgelements.Path):
                if len(element):
                    yield element, parents

            elif isinstance(element, svgelements.Shape):
                path = svgelements.Path(element)
                if len(path):
                    path.reify()
                    yield element, parents

    doc_info = DocumentInfo(parsed_svg.width, parsed_svg.height)  # type: ignore
    return ItemsFromDocument(elements(), doc_info)


def resolve_path(path: Union[pathlib.Path, str]) -> str:
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)
    return str(path.expanduser().resolve())


def _path_from_SvgPathLike(path: SvgPathLike) -> svgelements.Path:
    if isinstance(path, svgelements.Path):
        return path

    if not isinstance(path, str):
        path = format_svg_path(path)

    try:
        return svgelements.Path(str(path))
    except Exception:
        # TODO proper syntax error, would need to come from within svgpathelements
        raise ValueError(f"could not make svg path from: {path!r}")


def _continuous_subpaths(
    path: svgelements.Path,
) -> Iterator[tuple[svgelements.Path, bool]]:
    for subpath in path.as_subpaths():
        if subpath:
            is_closed = isinstance(subpath[-1], svgelements.Close)
            yield svgelements.Path(subpath), is_closed
