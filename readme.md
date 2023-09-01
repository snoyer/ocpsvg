# `ocpsvg`: `OCP` <-> SVG

- works at the [`OCP`](https://github.com/CadQuery/OCP) level
- uses [`svgpathtools`](https://github.com/mathandy/svgpathtools) to convert SVG path strings to and from `TopoDS_Edge`, `TopoDS_Wire`, and `TopoDS_Face` objects
- uses [`svgelements`](https://github.com/meerk40t/svgelements) to import `TopoDS_Wire` and `TopoDS_Face` objects from an SVG document
- can be used to add SVG functionality (import and export) to higher level API
