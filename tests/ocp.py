from OCP.BRepGProp import BRepGProp, BRepGProp_Face
from OCP.BRepTools import BRepTools
from OCP.gp import gp_Pnt, gp_Vec
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Face


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
