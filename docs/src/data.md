# Data, geometry, and bins

## Array layouts

| Input | Shape |
|---|---|
| Flat point positions | `(D, N)` |
| One vector field | `(D, N)` |
| Shared-position snapshots | `x: (D, N)`, `u: (D, N, batch...)` |
| Varying-position snapshots | `x` and `u`: `(D, N, batch...)` |
| Gridded vector field | `(components, grid_axes...)` |

`MultiFields.Fields(vectors=(u, v), scalars=(θ,))` groups fields sampled at the same positions. Mixed and cross-field operators select fields from that container. Tensor results place component axes before histogram and batch axes; see [result containers](api/results.md).

Unstructured point lists must contain valid coordinates and field values. Remove invalid observations before passing them to the package. Grid calculations preserve grid shape and omit pairs that touch masked or invalid cells.

## Geometry

Flat calculations use the coordinate metric and vector dimension. Spherical calculations use longitude/latitude coordinates and transport vectors into a pair frame before evaluating increments. Choose the spherical metric and radius explicitly; angular separation and physical distance have different units.

Coincident points and antipodal spherical points require care when an operator needs a direction. The geometry's pair-frame validity determines whether that pair contributes. [Mathematical definitions](theory.md) describes orientation and transverse conventions.

## Bins

A distance bin contains separations in `(left, right]`. Supply strictly increasing edges. `BinEdges` represents arbitrary edges; `LinearBinEdges` and `LogBinEdges` provide specialized regular grids. `InfPaddedBinEdges` adds unbounded end intervals. `midpoints` gives representative plotting coordinates; a result stores its edges, not those plotting coordinates.

Use `LinearBinEdges(first_edge, last_edge, n_edges)` or
`LogBinEdges(first_edge, last_edge, n_edges)` to construct regular grids.
`LogBinEdges_from_log_edges(range(...))` specifies a grid in logarithmic coordinates.
A vector of edges is a `BinEdges(vector)`, which keeps the supplied values. Every
route and backend places a separation `r` in bin `searchsortedfirst(edges, r) - 1`
of the edges as constructed; [Bin edges and digitize](digitize.md) gives the
definition of each grid's edges.

Joint calculations have a distance axis and either an operator-value axis or a separation-angle axis. These statistics have different meanings even when their array shapes match.

## Weights and masks

A point or cell weight `wᵢ` contributes pair mass `wᵢwⱼ`. Both the accumulated operator value and its normalization use that mass. Weighted counts require a floating-point type. For area or volume weighting on a grid, use `cell_measure(grid)`.

A mask removes invalid pairs; it does not impute observations. Interpreting the remaining pair averages as population statistics requires a sampling assumption.
