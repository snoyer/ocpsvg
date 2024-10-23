# `ocpsvg`: `OCP` ⟷ SVG

![Hidden lines rendering of faces imported from an `.svg` file](examples/svg-logo-hlr1.svg)
![Hidden lines rendering of solids extruded an debossed from an `.svg` file](examples/svg-logo-hlr2.svg)

- works at the [`OCP`](https://github.com/CadQuery/OCP) level
- uses [`svgelements`](https://github.com/meerk40t/svgelements) to:
  - convert SVG path strings to and from `TopoDS_Edge`, `TopoDS_Wire`, and `TopoDS_Face` objects
  - import `TopoDS_Wire` and `TopoDS_Face` objects from an SVG document
- can be used to add SVG functionality (import and export) to higher level API
