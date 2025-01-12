from typing import Iterable

from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, TopoDS_Wire
from OCP.TopTools import TopTools_ListOfShape


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


def is_valid(shape: TopoDS_Shape):
    check = BRepCheck_Analyzer(shape)
    check.SetParallel(True)
    return check.IsValid()

def wire_via_BRepBuilderAPI(edges: Iterable[TopoDS_Edge]) -> TopoDS_Wire:
    """Make a wire using `BRepBuilderAPI_MakeWire.Wire`"""
    makewire = BRepBuilderAPI_MakeWire()
    edge_list = TopTools_ListOfShape()
    for edge in edges:
        edge_list.Append(edge)
    makewire.Add(edge_list)
    makewire.Build()  # type: ignore
    return makewire.Wire()

