from typing import Iterable

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Wire

from ocpsvg.ocp import closed_wire


def face_area(face: TopoDS_Face):
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, properties)

    return properties.Mass()


def face_normal(face: TopoDS_Face) -> gp_Vec:
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    return face_normal_at_uv(face, (u0 + u1) / 2, (v0 + v1) / 2)


def face_normal_at_uv(face: TopoDS_Face, u: float, v: float) -> gp_Vec:
    gp_pnt = gp_Pnt()
    normal = gp_Vec()
    BRepGProp_Face(face).Normal(u, v, gp_pnt, normal)
    return normal


def wire_from_edges(
    edges: Iterable[TopoDS_Edge], *, closed: bool = False
) -> TopoDS_Wire:
    """Make a single wire from edges using `BRepBuilderAPI_MakeWire`."""
    builder = BRepBuilderAPI_MakeWire()
    for edge in edges:
        builder.Add(edge)

    builder.Build()  # type: ignore
    if builder.IsDone():
        wire = builder.Wire()
        return closed_wire(wire) if closed else wire
    else:
        raise ValueError("could not build wire from edges")
