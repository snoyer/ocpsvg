# `ocpsvg`: `OCP` <-> SVG

- works at the OCP level
- uses `svgpathtools` to convert SVG path strings to and from `TopoDS_Edge`, `TopoDS_Wire`, and `TopoDS_Face` objects
- uses `svgelements` to import `TopoDS_Wire` and `TopoDS_Face` objects from an SVG document
- can be used to add SVG functionality (import and export) to higher level API
